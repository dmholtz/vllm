// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! OpenAI-compatible `/v1/embeddings`, mirroring Python's `ServingEmbedding`
//! and `EmbedIOProcessor` (`vllm/entrypoints/pooling/embed/`).

mod types;

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use vllm_engine_core_client::protocol::pooling::PoolingFrontendConfig;
use vllm_llm::{EncodeOutput, PoolingParams, PoolingTask};
use vllm_text::{PoolingTokenization, Prompt, TextEncodeRequest};

use self::types::{
    EmbeddingData, EmbeddingPrompts, EmbeddingRequest, EmbeddingResponse, EmbeddingResponseData,
    Padding,
};
use super::utils::validated_json::ValidatedJson;
use crate::error::{ApiError, bail_invalid_request, server_error, text_submit_error};
use crate::routes::pooling::check_truncation_limit;
use crate::routes::pooling_chat::render_conversations;
use crate::routes::pooling_encoding::{
    BinaryItem, BinaryResponseInfo, EncodingFormat, PoolingUsage, binary_response, encode_base64,
};
use crate::state::AppState;
use crate::utils::{ResolvedRequestContext, resolve_request_context, unix_timestamp};

/// Serve one OpenAI-compatible embeddings request.
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<EmbeddingRequest>,
) -> Response {
    match create_embedding(&state, &headers, body).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

/// One embedding and the number of prompt tokens it consumed.
struct Embedding {
    values: Vec<f32>,
    shape: Vec<usize>,
    prompt_tokens: usize,
}

async fn create_embedding(
    state: &AppState,
    headers: &HeaderMap,
    mut body: EmbeddingRequest,
) -> Result<Response, ApiError> {
    body.validate_compat()?;

    let requested_model = body.model.as_deref().filter(|model| !model.is_empty());
    let resolution = state.resolve_model_with_loras(requested_model).await;
    if let Some(model) = requested_model
        && !resolution.model_names.iter().any(|name| name == model)
    {
        return Err(ApiError::model_not_found(model.to_owned()));
    }
    let response_model = (resolution.lora_request.as_ref())
        .map(|request| request.lora_name.clone())
        .unwrap_or_else(|| resolution.model_names.first().cloned().unwrap_or_default());
    let ctx = resolve_request_context(headers, body.request_id.as_deref());
    let response_id = format!("embd-{}", ctx.request_id);

    let config = state
        .serving_tasks()
        .and_then(|tasks| tasks.pooling.clone())
        .unwrap_or_default();
    let text = state.chat.text();
    let max_model_len = text.request_processor().max_model_len();
    check_truncation_limit(body.truncate_prompt_tokens, max_model_len)?;
    let is_chat = body.is_chat();
    let tokenization = embedding_tokenization(&body, &config, max_model_len, is_chat);

    // Render and tokenize every input before submitting any of them.
    let prompts = match body.take_prompts()? {
        EmbeddingPrompts::Completion(prompts) => prompts,
        EmbeddingPrompts::Chat(conversations) => {
            render_conversations(
                state,
                &body.chat_render_options(),
                &response_id,
                conversations,
            )
            .await?
        }
    };
    let processor = text.request_processor();
    let prompt_token_ids = prompts
        .into_iter()
        .map(|prompt| processor.tokenize_pooling_prompt(prompt, &tokenization))
        .collect::<vllm_text::Result<Vec<_>>>()
        .map_err(|error| text_submit_error("failed to tokenize embedding input", error))?;

    let pooling_params = PoolingParams {
        dimensions: body.dimensions,
        use_activation: body.use_activation,
        ..Default::default()
    };
    let submission = Submission {
        response_id: &response_id,
        ctx: &ctx,
        body: &body,
        pooling_params: &pooling_params,
        lora_request: resolution.lora_request.as_ref(),
        max_model_len,
    };
    let embeddings = if config.enable_chunked_processing {
        // The pooler default applies when the request leaves it unset.
        let use_activation = body.use_activation.or(config.use_activation).unwrap_or(true);
        embed_chunked(text, &submission, prompt_token_ids, use_activation).await?
    } else {
        embed_whole(text, &submission, prompt_token_ids).await?
    };

    build_response(&body, &response_id, &response_model, embeddings)
}

/// Python `EmbeddingTokenizeParamsMixin.build_tok_params`.
fn embedding_tokenization(
    body: &EmbeddingRequest,
    config: &PoolingFrontendConfig,
    max_model_len: u32,
    is_chat: bool,
) -> PoolingTokenization {
    // With chunked processing, inputs may exceed `max_model_len`; they are
    // limited only by `max_embed_len`, if set.
    let max_total_tokens = if config.enable_chunked_processing {
        config.max_embed_len
    } else {
        Some(config.max_embed_len.unwrap_or(max_model_len))
    };
    let pad_prompt_tokens = match body.padding {
        Some(Padding::MaxLength) if config.enable_chunked_processing => {
            Some(max_model_len.min(config.max_embed_len.unwrap_or(max_model_len)))
        }
        Some(Padding::MaxLength) => max_total_tokens,
        Some(Padding::DoNotPad) | None => None,
    };
    PoolingTokenization {
        max_total_tokens,
        truncate_prompt_tokens: body.truncate_prompt_tokens,
        truncation_side: body.truncation_side,
        pad_prompt_tokens,
        do_lower_case: config.do_lower_case,
        // Chat templates add their own special tokens.
        add_special_tokens: body.add_special_tokens.unwrap_or(!is_chat),
    }
}

/// Per-request fields shared by every engine request of one embeddings call.
struct Submission<'a> {
    response_id: &'a str,
    ctx: &'a ResolvedRequestContext,
    body: &'a EmbeddingRequest,
    pooling_params: &'a PoolingParams,
    lora_request: Option<&'a vllm_engine_core_client::protocol::lora::LoraRequest>,
    max_model_len: u32,
}

impl Submission<'_> {
    /// Build one engine request for already-tokenized prompt tokens.
    fn request(&self, request_id: String, prompt_token_ids: Vec<u32>) -> TextEncodeRequest {
        TextEncodeRequest {
            request_id,
            prompt: Prompt::TokenIds(prompt_token_ids),
            task: PoolingTask::Embed,
            pooling_params: self.pooling_params.clone(),
            // Tokens were prepared already; only the context window applies.
            tokenization: PoolingTokenization::new(Some(self.max_model_len), false),
            arrival_time: Some(vllm_llm::current_unix_timestamp_secs()),
            cache_salt: self.body.cache_salt.clone(),
            trace_headers: None,
            priority: self.ctx.priority.or(self.body.priority).unwrap_or(0),
            data_parallel_rank: self.ctx.data_parallel_rank,
            session_id: self.ctx.session_id.clone(),
            lora_request: self.lora_request.cloned(),
        }
    }
}

fn output_vector(output: EncodeOutput) -> Result<(Vec<f32>, Vec<usize>), ApiError> {
    if output.output.shape.len() != 1 {
        return Err(server_error!(
            "embedding request `{}` expected a rank-1 pooling tensor, got shape {:?}",
            output.request_id,
            output.output.shape
        ));
    }
    Ok((output.output.data, output.output.shape))
}

/// Embed each prompt with one engine request.
async fn embed_whole(
    text: &vllm_text::TextLlm,
    submission: &Submission<'_>,
    prompt_token_ids: Vec<Vec<u32>>,
) -> Result<Vec<Embedding>, ApiError> {
    let requests = prompt_token_ids
        .into_iter()
        .enumerate()
        .map(|(index, tokens)| {
            submission.request(format!("{}-{index}", submission.response_id), tokens)
        })
        .collect();
    let outputs = text
        .encode_batch(requests)
        .await
        .map_err(|error| text_submit_error("failed to submit embedding request", error))?;
    outputs
        .into_iter()
        .map(|output| {
            let prompt_tokens = output.prompt_token_ids.len();
            let (values, shape) = output_vector(output)?;
            Ok(Embedding {
                values,
                shape,
                prompt_tokens,
            })
        })
        .collect()
}

/// Python `EmbedIOProcessor.maybe_pre_process_chunked` and
/// `_post_process_chunked`: split every prompt into `max_model_len` chunks,
/// embed all chunks concurrently and combine each prompt's chunk embeddings
/// into their token-weighted mean.
async fn embed_chunked(
    text: &vllm_text::TextLlm,
    submission: &Submission<'_>,
    prompt_token_ids: Vec<Vec<u32>>,
    use_activation: bool,
) -> Result<Vec<Embedding>, ApiError> {
    let chunk_size = submission.max_model_len.max(1) as usize;
    let mut requests = Vec::new();
    let mut chunk_prompt_index = Vec::new();
    for (prompt_index, tokens) in prompt_token_ids.iter().enumerate() {
        if tokens.is_empty() {
            bail_invalid_request!(param = "input", "input must not contain empty prompts.");
        }
        for (chunk_index, chunk) in tokens.chunks(chunk_size).enumerate() {
            requests.push(submission.request(
                format!(
                    "{}-prompt-{prompt_index}-chunk-{chunk_index}",
                    submission.response_id
                ),
                chunk.to_vec(),
            ));
            chunk_prompt_index.push(prompt_index);
        }
    }
    let outputs = text
        .encode_batch(requests)
        .await
        .map_err(|error| text_submit_error("failed to submit embedding request", error))?;

    let mut chunks_per_prompt = vec![Vec::new(); prompt_token_ids.len()];
    for (output, prompt_index) in outputs.into_iter().zip(chunk_prompt_index) {
        let weight = output.prompt_token_ids.len();
        let (values, _) = output_vector(output)?;
        chunks_per_prompt[prompt_index].push((weight, values));
    }
    chunks_per_prompt
        .into_iter()
        .zip(prompt_token_ids)
        .map(|(chunks, tokens)| {
            let values = aggregate_chunks(&chunks, use_activation)?;
            Ok(Embedding {
                shape: vec![values.len()],
                values,
                prompt_tokens: tokens.len(),
            })
        })
        .collect()
}

/// Token-weighted mean of chunk embeddings, renormalized when activation is
/// on, the prompt spans several chunks and every chunk embedding was
/// L2-normalized (custom activations need not normalize).
fn aggregate_chunks(
    chunks: &[(usize, Vec<f32>)],
    use_activation: bool,
) -> Result<Vec<f32>, ApiError> {
    let Some((_, first)) = chunks.first() else {
        return Err(server_error!("failed to aggregate embedding chunks"));
    };
    let mut weighted_sum = vec![0f32; first.len()];
    let mut total_weight = 0usize;
    let mut all_chunks_l2_normalized = true;
    for (weight, values) in chunks {
        if values.len() != weighted_sum.len() {
            return Err(server_error!("embedding chunks have different dimensions"));
        }
        if use_activation {
            // `torch.allclose(norm, 1, rtol=1e-2, atol=1e-5)`.
            let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
            all_chunks_l2_normalized &= (norm - 1.0).abs() <= 1e-5 + 1e-2;
        }
        for (sum, value) in weighted_sum.iter_mut().zip(values) {
            *sum += value * *weight as f32;
        }
        total_weight += weight;
    }
    if total_weight == 0 {
        return Err(server_error!("failed to aggregate embedding chunks"));
    }
    let mut mean: Vec<f32> =
        weighted_sum.into_iter().map(|sum| sum / total_weight as f32).collect();
    if use_activation && chunks.len() > 1 && all_chunks_l2_normalized {
        // `torch.nn.functional.normalize(x, dim=-1)` with `eps=1e-12`.
        let norm = mean.iter().map(|value| value * value).sum::<f32>().sqrt().max(1e-12);
        mean.iter_mut().for_each(|value| *value /= norm);
    }
    Ok(mean)
}

fn build_response(
    body: &EmbeddingRequest,
    response_id: &str,
    response_model: &str,
    embeddings: Vec<Embedding>,
) -> Result<Response, ApiError> {
    let prompt_tokens = embeddings.iter().map(|embedding| embedding.prompt_tokens).sum();
    let created = unix_timestamp();

    if body.encoding_format.is_binary() {
        let items = embeddings
            .iter()
            .map(|embedding| BinaryItem {
                values: &embedding.values,
                shape: &embedding.shape,
            })
            .collect::<Vec<_>>();
        return binary_response(
            &items,
            BinaryResponseInfo {
                id: response_id,
                created,
                model: response_model,
                prompt_tokens,
            },
            body.encoding_format,
            body.embed_dtype,
            body.endianness,
        );
    }

    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingResponseData {
            index,
            object: "embedding",
            embedding: match body.encoding_format {
                EncodingFormat::Base64 => EmbeddingData::Base64(encode_base64(
                    &embedding.values,
                    body.embed_dtype,
                    body.endianness,
                )),
                _ => EmbeddingData::Float(embedding.values),
            },
        })
        .collect();
    Ok(Json(EmbeddingResponse {
        id: response_id.to_owned(),
        object: "list",
        created,
        model: response_model.to_owned(),
        data,
        usage: PoolingUsage::new(prompt_tokens),
    })
    .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(values: &[f32]) -> f32 {
        values.iter().map(|value| value * value).sum::<f32>().sqrt()
    }

    #[test]
    fn chunk_aggregation_is_a_token_weighted_mean() {
        let chunks = [(3, vec![1.0, 0.0]), (1, vec![0.0, 1.0])];
        let mean = aggregate_chunks(&chunks, false).unwrap();
        assert_eq!(mean, vec![0.75, 0.25]);

        // Normalized chunks with activation are renormalized after averaging.
        let normalized = aggregate_chunks(&chunks, true).unwrap();
        assert!((norm(&normalized) - 1.0).abs() < 1e-6);
        assert!((normalized[0] / normalized[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn chunk_aggregation_skips_renormalization_when_not_applicable() {
        // A single chunk is returned as is.
        let single = aggregate_chunks(&[(5, vec![0.6, 0.8])], true).unwrap();
        assert_eq!(single, vec![0.6, 0.8]);
        // Unnormalized chunks (custom activation) are only averaged.
        let chunks = [(1, vec![2.0, 0.0]), (1, vec![0.0, 2.0])];
        assert_eq!(aggregate_chunks(&chunks, true).unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn tokenization_follows_python_embedding_limits() {
        let body: EmbeddingRequest =
            serde_json::from_value(serde_json::json!({"input": "a", "padding": "max_length"}))
                .unwrap();
        let mut config = PoolingFrontendConfig::default();
        let plain = embedding_tokenization(&body, &config, 512, false);
        assert_eq!(plain.max_total_tokens, Some(512));
        assert_eq!(plain.pad_prompt_tokens, Some(512));
        assert!(plain.add_special_tokens);

        config.max_embed_len = Some(256);
        assert_eq!(
            embedding_tokenization(&body, &config, 512, false).max_total_tokens,
            Some(256)
        );

        config.enable_chunked_processing = true;
        config.max_embed_len = Some(4096);
        let chunked = embedding_tokenization(&body, &config, 512, true);
        assert_eq!(chunked.max_total_tokens, Some(4096));
        assert_eq!(chunked.pad_prompt_tokens, Some(512));
        assert!(!chunked.add_special_tokens);

        config.max_embed_len = None;
        assert_eq!(
            embedding_tokenization(&body, &config, 512, false).max_total_tokens,
            None
        );
    }
}
