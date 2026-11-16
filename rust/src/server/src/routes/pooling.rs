// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `/pooling` and `/classify`, mirroring Python's `ServingPooling` and
//! `ServingClassification` (`vllm/entrypoints/pooling/`).

mod types;

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use vllm_llm::{EncodeOutput, EngineTask, PoolingParams, PoolingTask};
use vllm_text::{PoolingTokenization, TextEncodeRequest};

use self::types::{Padding, PoolingRequest};
use crate::error::{
    ApiError, bail_invalid_request, invalid_request, server_error, text_submit_error,
};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::routes::pooling_chat::render_conversations;
use crate::routes::pooling_encoding::{
    BinaryItem, BinaryResponseInfo, EncodingFormat, PoolingUsage, binary_response, encode_base64,
};
use crate::state::AppState;
use crate::utils::{resolve_request_context, unix_timestamp};

#[derive(Clone, Copy)]
enum Endpoint {
    Pooling,
    Classify,
}

pub async fn pooling(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<PoolingRequest>,
) -> Response {
    handle_request(&state, &headers, body, Endpoint::Pooling).await
}

pub async fn classify(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<PoolingRequest>,
) -> Response {
    handle_request(&state, &headers, body, Endpoint::Classify).await
}

async fn handle_request(
    state: &AppState,
    headers: &HeaderMap,
    body: PoolingRequest,
    endpoint: Endpoint,
) -> Response {
    let prepared = match prepare_request(state, headers, body, endpoint).await {
        Ok(prepared) => prepared,
        Err(error) => return error.into_response(),
    };
    match run_pooling(state, prepared).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

struct PreparedRequest {
    response_id: String,
    response_model: String,
    body: PoolingRequest,
    endpoint: Endpoint,
    requests: Vec<TextEncodeRequest>,
}

/// Python `PoolingBaseServing._validate_request`, which runs before
/// tokenization for every pooling endpoint.
pub(crate) fn check_truncation_limit(
    truncate_prompt_tokens: Option<i64>,
    max_model_len: u32,
) -> Result<(), ApiError> {
    if truncate_prompt_tokens.is_some_and(|value| value > i64::from(max_model_len)) {
        bail_invalid_request!(
            param = "truncate_prompt_tokens",
            "truncate_prompt_tokens value is greater than max_model_len. Please request a \
             smaller truncation size."
        );
    }
    Ok(())
}

/// Wire literal of a task, as Python prints it in error messages.
fn task_literal(task: impl serde::Serialize) -> String {
    serde_json::to_value(task)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_default()
}

/// Resolve the pooling task like Python `ServingPooling._verify_pooling_task`.
async fn resolve_task(
    state: &AppState,
    body: &PoolingRequest,
    endpoint: Endpoint,
) -> Result<PoolingTask, ApiError> {
    let requested = body.task()?;
    let supported = state
        .chat
        .text()
        .supported_tasks()
        .await
        .map_err(|error| text_submit_error("failed to discover pooling tasks", error))?;
    let model_task = state.serving_tasks().and_then(|tasks| tasks.pooling_task()).or_else(|| {
        // States built without an engine handshake: the priority order of
        // Python `ModelConfig.get_pooling_task`.
        [
            PoolingTask::EmbedAndTokenClassify,
            PoolingTask::Embed,
            PoolingTask::Classify,
            PoolingTask::TokenEmbed,
            PoolingTask::TokenClassify,
            PoolingTask::Plugin,
        ]
        .into_iter()
        .find(|task| supported.contains(&EngineTask::Pooling(*task)))
    });

    let task = match endpoint {
        // Python's `ClassificationRequest` has no `task` field; `/classify`
        // always classifies.
        Endpoint::Classify => PoolingTask::Classify,
        Endpoint::Pooling => {
            if body.dimensions.is_some() {
                bail_invalid_request!(
                    param = "dimensions",
                    "dimensions is currently not supported"
                );
            }
            let task = requested.or(model_task).ok_or_else(|| {
                invalid_request!(param = "task", "The model does not support a pooling task.")
            })?;
            if Some(task) != model_task {
                if supported.contains(&EngineTask::Pooling(task)) {
                    bail_invalid_request!(
                        param = "task",
                        "Try switching the model's pooling_task via --pooler-config.task {}.",
                        task_literal(task)
                    );
                }
                // Python formats the supported tasks as a list repr.
                let supported_repr = supported
                    .iter()
                    .map(|task| format!("'{}'", task_literal(task)))
                    .collect::<Vec<_>>()
                    .join(", ");
                bail_invalid_request!(
                    param = "task",
                    "Unsupported task: '{}' Supported tasks: [{supported_repr}]",
                    task_literal(task),
                );
            }
            task
        }
    };
    if !supported.contains(&EngineTask::Pooling(task)) {
        bail_invalid_request!(param = "task", "model does not support {task:?}");
    }
    match task {
        // TODO: integrate IOProcessor plugins.
        PoolingTask::Plugin => bail_invalid_request!(
            param = "task",
            "IOProcessor plugins are not supported by the Rust frontend yet."
        ),
        PoolingTask::EmbedAndTokenClassify => bail_invalid_request!(
            param = "task",
            "The `embed&token_classify` task is not supported by the online pooling API."
        ),
        task => Ok(task),
    }
}

async fn prepare_request(
    state: &AppState,
    headers: &HeaderMap,
    mut body: PoolingRequest,
    endpoint: Endpoint,
) -> Result<PreparedRequest, ApiError> {
    body.validate_options()?;
    let max_model_len = state.chat.text().request_processor().max_model_len();
    check_truncation_limit(body.truncate_prompt_tokens, max_model_len)?;
    let model = body.model.as_deref().filter(|model| !model.is_empty());
    let resolution = state.resolve_model_with_loras(model).await;
    if let Some(model) = model
        && !resolution.model_names.iter().any(|name| name == model)
    {
        return Err(ApiError::model_not_found(model.to_owned()));
    }
    let task = resolve_task(state, &body, endpoint).await?;

    let prefix = match endpoint {
        Endpoint::Pooling => "pool",
        Endpoint::Classify => "classify",
    };
    let ctx = resolve_request_context(headers, body.request_id.as_deref());
    let id = format!("{prefix}-{}", ctx.request_id);
    let response_model = (resolution.lora_request.as_ref())
        .map(|lora| lora.lora_name.clone())
        .unwrap_or_else(|| resolution.model_names.first().cloned().unwrap_or_default());

    let is_chat = body.messages.is_some();
    let prompts = match (body.messages.take(), body.input.take()) {
        (Some(messages), _) => {
            render_conversations(state, &body.chat_render_options(), &id, vec![messages]).await?
        }
        (None, Some(input)) => input.into_prompts(),
        (None, None) => unreachable!("validated by `validate_options`"),
    };
    if prompts.is_empty() {
        bail_invalid_request!(param = "input", "input must contain at least one prompt");
    }

    // Python `FixedMaxLenTokenizeParamsMixin`: limited to `max_model_len`.
    let tokenization = PoolingTokenization {
        truncate_prompt_tokens: body.truncate_prompt_tokens,
        truncation_side: body.truncation_side,
        pad_prompt_tokens: (body.padding == Some(Padding::MaxLength)).then_some(max_model_len),
        do_lower_case: state
            .serving_tasks()
            .and_then(|tasks| tasks.pooling.as_ref())
            .is_some_and(|config| config.do_lower_case),
        ..PoolingTokenization::new(
            Some(max_model_len),
            // Chat templates add their own special tokens.
            body.add_special_tokens.unwrap_or(!is_chat),
        )
    };
    let pooling_params = PoolingParams {
        // Python's `ClassificationRequest` has no `dimensions`; it is ignored.
        dimensions: body.dimensions.filter(|_| matches!(endpoint, Endpoint::Pooling)),
        use_activation: body.use_activation,
        ..Default::default()
    };
    let requests = prompts
        .into_iter()
        .enumerate()
        .map(|(index, prompt)| TextEncodeRequest {
            prompt,
            tokenization: tokenization.clone(),
            request_id: format!("{id}-{index}"),
            task,
            pooling_params: pooling_params.clone(),
            arrival_time: Some(vllm_llm::current_unix_timestamp_secs()),
            cache_salt: body.cache_salt.clone(),
            trace_headers: None,
            priority: ctx.priority.unwrap_or(body.priority),
            data_parallel_rank: ctx.data_parallel_rank,
            session_id: ctx.session_id.clone(),
            lora_request: resolution.lora_request.clone(),
        })
        .collect();
    Ok(PreparedRequest {
        response_id: id,
        response_model,
        body,
        endpoint,
        requests,
    })
}

async fn run_pooling(state: &AppState, prepared: PreparedRequest) -> Result<Response, ApiError> {
    let text = state.chat.text();
    let outputs = text
        .encode_batch(prepared.requests)
        .await
        .map_err(|error| text_submit_error("failed to encode pooling batch", error))?;
    let prompt_tokens = outputs.iter().map(|output| output.prompt_token_ids.len()).sum();
    let created = unix_timestamp();
    let body = &prepared.body;

    let data = match prepared.endpoint {
        Endpoint::Pooling if body.encoding_format.is_binary() => {
            let items = outputs
                .iter()
                .map(|output| BinaryItem {
                    values: &output.output.data,
                    shape: &output.output.shape,
                })
                .collect::<Vec<_>>();
            return binary_response(
                &items,
                BinaryResponseInfo {
                    id: &prepared.response_id,
                    created,
                    model: &prepared.response_model,
                    prompt_tokens,
                },
                body.encoding_format,
                body.embed_dtype,
                body.endianness,
            );
        }
        Endpoint::Pooling => outputs
            .iter()
            .enumerate()
            .map(|(index, output)| {
                let data = match body.encoding_format {
                    EncodingFormat::Base64 => {
                        json!(encode_base64(
                            &output.output.data,
                            body.embed_dtype,
                            body.endianness
                        ))
                    }
                    _ => tensor_json(output)?,
                };
                Ok(json!({"index": index, "object": "pooling", "data": data}))
            })
            .collect::<Result<Vec<_>, ApiError>>()?,
        Endpoint::Classify => {
            let pooling_config = state.serving_tasks().and_then(|tasks| tasks.pooling.as_ref());
            outputs
                .iter()
                .enumerate()
                .map(|(index, output)| {
                    let probs = vector(output)?;
                    // `np.argmax`: the first maximum wins.
                    let predicted = probs
                        .iter()
                        .enumerate()
                        .reduce(|best, item| if item.1 > best.1 { item } else { best })
                        .ok_or_else(|| server_error!("classifier returned no classes"))?
                        .0;
                    // Labels resolved by engine-core honor `--hf-overrides`.
                    let label = match pooling_config {
                        Some(config) => config.label(predicted),
                        None => text.classification_label(predicted),
                    };
                    Ok(json!({
                        "index": index,
                        "label": label,
                        "probs": probs,
                        "num_classes": probs.len(),
                    }))
                })
                .collect::<Result<Vec<_>, ApiError>>()?
        }
    };
    Ok(Json(json!({
        "id": prepared.response_id,
        "object": "list",
        "created": created,
        "model": prepared.response_model,
        "data": data,
        "usage": PoolingUsage::new(prompt_tokens),
    }))
    .into_response())
}

fn vector(output: &EncodeOutput) -> Result<&[f32], ApiError> {
    if output.output.shape != [output.output.data.len()] {
        return Err(server_error!(
            "expected a pooling vector, got shape {:?}",
            output.output.shape
        ));
    }
    Ok(&output.output.data)
}

fn tensor_json(output: &EncodeOutput) -> Result<Value, ApiError> {
    let tensor = &output.output;
    match tensor.shape.as_slice() {
        [n] if *n == tensor.data.len() => Ok(json!(tensor.data)),
        [rows, columns] if rows.checked_mul(*columns) == Some(tensor.data.len()) => {
            Ok(Value::Array(
                (0..*rows)
                    .map(|row| json!(&tensor.data[row * columns..(row + 1) * columns]))
                    .collect(),
            ))
        }
        _ => Err(server_error!(
            "unsupported pooling tensor shape {:?}",
            tensor.shape
        )),
    }
}
