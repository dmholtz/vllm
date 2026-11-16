// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;
use vllm_text::{Prompt, TruncationSide};

use crate::error::{ApiError, bail_invalid_request};
use crate::routes::openai::utils::types::{ChatMessage, Normalizable};
use crate::routes::pooling_chat::ChatRenderOptions;
use crate::routes::pooling_encoding::{EmbedDType, EncodingFormat, Endianness, PoolingUsage};

/// Text, token-ID or chat-conversation input of the embeddings endpoint.
///
/// Python accepts a chat conversation in `input` as well as in `messages`
/// (`EmbeddingChatInputRequest` / `EmbeddingBatchChatInputRequest`).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum EmbeddingInput {
    TokenIds(Vec<u32>),
    TokenIdBatch(Vec<Vec<u32>>),
    Text(String),
    TextBatch(Vec<String>),
    Conversation(Vec<ChatMessage>),
    ConversationBatch(Vec<Vec<ChatMessage>>),
}

/// Chat conversation(s) in the `messages` field.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum EmbeddingMessages {
    Conversation(Vec<ChatMessage>),
    ConversationBatch(Vec<Vec<ChatMessage>>),
}

/// Prompts of one embeddings request after resolving the request variant.
pub(crate) enum EmbeddingPrompts {
    Completion(Vec<Prompt>),
    Chat(Vec<Vec<ChatMessage>>),
}

/// Python `padding` tokenization option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Padding {
    /// Pad to the maximum input length.
    MaxLength,
    /// Leave the prompt as is.
    DoNotPad,
}

/// OpenAI-compatible embeddings request.
///
/// Covers every variant of Python vLLM's `EmbeddingRequest` union: completion
/// inputs (`EmbeddingCompletionRequest`) and chat conversations in `messages`
/// or `input`, single or batched (`EmbeddingChatRequest`,
/// `EmbeddingBatchChatRequest`, `EmbeddingChatInputRequest`,
/// `EmbeddingBatchChatInputRequest`).
#[derive(Debug, Clone, Deserialize, Validate)]
pub(crate) struct EmbeddingRequest {
    /// ID of the model to use. An omitted or empty value selects the default.
    pub model: Option<String>,
    /// One text/token-ID prompt, a batch of homogeneous prompts, or chat
    /// conversation(s).
    pub input: Option<EmbeddingInput>,
    /// Chat conversation(s) to render with the chat template.
    pub messages: Option<EmbeddingMessages>,
    /// Response encoding.
    #[serde(default)]
    pub encoding_format: EncodingFormat,
    /// Numeric type of `base64` and `bytes` outputs.
    #[serde(default)]
    pub embed_dtype: EmbedDType,
    /// Byte order of `base64` and `bytes` outputs.
    #[serde(default)]
    pub endianness: Endianness,
    /// Reduce the dimensions of embeddings for models that support
    /// Matryoshka representation. Engine-core validates the value.
    pub dimensions: Option<u32>,
    /// Whether to apply activation to pooler outputs. `None` uses the pooler's
    /// model-aware default.
    pub use_activation: Option<bool>,
    /// Whether tokenization adds special tokens such as BOS. Defaults to
    /// `true` for completion inputs and `false` for chat inputs, whose
    /// template adds them.
    pub add_special_tokens: Option<bool>,
    /// Number of prompt tokens to retain. `-1` keeps the full input budget;
    /// `None` leaves the prompt untruncated.
    pub truncate_prompt_tokens: Option<i64>,
    /// Side from which excess prompt tokens are discarded.
    pub truncation_side: Option<TruncationSide>,
    /// Pad the prompt to the maximum input length.
    pub padding: Option<Padding>,
    /// Request ID used throughout inference and returned in the response.
    pub request_id: Option<String>,
    /// Scheduling priority; lower values are handled earlier.
    pub priority: Option<i32>,
    /// Random salt used to isolate prefix-cache entries across users.
    pub cache_salt: Option<String>,
    /// End-user identifier; accepted and ignored, like Python.
    #[allow(dead_code)]
    pub user: Option<String>,
    /// Chat inputs: start a new assistant turn after the conversation.
    #[serde(default)]
    pub add_generation_prompt: bool,
    /// Chat inputs: leave the final message open-ended.
    #[serde(default)]
    pub continue_final_message: bool,
    /// Chat inputs: per-request chat template override.
    pub chat_template: Option<String>,
    /// Chat inputs: additional keyword arguments for the chat template.
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    /// Multimodal processor options; unsupported until multimodal pooling
    /// inputs land.
    pub mm_processor_kwargs: Option<Value>,
    /// Media IO options; unsupported until multimodal pooling inputs land.
    pub media_io_kwargs: Option<Value>,
    /// Fields not modeled above, checked for removed parameters.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Normalizable for EmbeddingRequest {}

impl EmbeddingRequest {
    /// Python's pydantic-level request checks.
    pub(crate) fn validate_compat(&self) -> Result<(), ApiError> {
        // `reject_removed_pooling_parameters`.
        if self.extra.contains_key("normalize") {
            bail_invalid_request!(
                param = "normalize",
                "Parameter `normalize` was removed; use `use_activation` instead."
            );
        }
        match self.extra.get("task").and_then(Value::as_str) {
            Some("score") => bail_invalid_request!(
                param = "task",
                "`score` task was removed; use `classify` instead."
            ),
            Some("encode") => bail_invalid_request!(
                param = "task",
                "`encode` task was removed; use `token_embed` or `token_classify` instead."
            ),
            _ => {}
        }
        if self.truncate_prompt_tokens.is_some_and(|value| value < -1) {
            bail_invalid_request!(
                param = "truncate_prompt_tokens",
                "truncate_prompt_tokens must be greater than or equal to -1."
            );
        }
        if self
            .cache_salt
            .as_deref()
            .is_some_and(|salt| !vllm_text::is_valid_cache_salt(salt))
        {
            bail_invalid_request!(
                param = "cache_salt",
                "Parameter 'cache_salt' must be a non-empty string of at most 128 characters \
                 and must not contain '@', '/', '\\\\', or NUL."
            );
        }
        if self.continue_final_message && self.add_generation_prompt {
            bail_invalid_request!(
                "Cannot set both `continue_final_message` and `add_generation_prompt` to True."
            );
        }
        // TODO: support multimodal pooling inputs.
        for (param, value) in [
            ("mm_processor_kwargs", &self.mm_processor_kwargs),
            ("media_io_kwargs", &self.media_io_kwargs),
        ] {
            if value.as_ref().is_some_and(|value| !value.is_null()) {
                bail_invalid_request!(
                    param = param,
                    "`{param}` is not supported by the Rust frontend yet."
                );
            }
        }
        Ok(())
    }

    /// Chat rendering options of this request.
    pub(crate) fn chat_render_options(&self) -> ChatRenderOptions {
        ChatRenderOptions {
            add_generation_prompt: self.add_generation_prompt,
            continue_final_message: self.continue_final_message,
            chat_template: self.chat_template.clone(),
            chat_template_kwargs: self.chat_template_kwargs.clone(),
        }
    }

    /// Whether this request carries chat conversations.
    pub(crate) fn is_chat(&self) -> bool {
        self.messages.is_some()
            || matches!(
                self.input,
                Some(EmbeddingInput::Conversation(_) | EmbeddingInput::ConversationBatch(_))
            )
    }

    /// Resolve the request variant into completion prompts or conversations.
    pub(crate) fn take_prompts(&mut self) -> Result<EmbeddingPrompts, ApiError> {
        let prompts = match (self.messages.take(), self.input.take()) {
            (Some(EmbeddingMessages::Conversation(messages)), _)
            | (None, Some(EmbeddingInput::Conversation(messages))) => {
                EmbeddingPrompts::Chat(vec![messages])
            }
            (Some(EmbeddingMessages::ConversationBatch(batch)), _)
            | (None, Some(EmbeddingInput::ConversationBatch(batch))) => {
                if batch.iter().any(Vec::is_empty) {
                    bail_invalid_request!(
                        param = "messages",
                        "Each conversation must contain at least one message."
                    );
                }
                EmbeddingPrompts::Chat(batch)
            }
            (None, Some(input)) => EmbeddingPrompts::Completion(match input {
                EmbeddingInput::TokenIds(token_ids) => vec![Prompt::TokenIds(token_ids)],
                EmbeddingInput::TokenIdBatch(batch) => {
                    batch.into_iter().map(Prompt::TokenIds).collect()
                }
                EmbeddingInput::Text(text) => vec![Prompt::Text(text)],
                EmbeddingInput::TextBatch(batch) => batch.into_iter().map(Prompt::Text).collect(),
                EmbeddingInput::Conversation(_) | EmbeddingInput::ConversationBatch(_) => {
                    unreachable!("handled above")
                }
            }),
            (None, None) => bail_invalid_request!(
                param = "input",
                "Either `input` or `messages` must be provided."
            ),
        };
        let is_empty = match &prompts {
            EmbeddingPrompts::Completion(prompts) => prompts.is_empty(),
            EmbeddingPrompts::Chat(conversations) => {
                conversations.is_empty() || conversations.iter().all(Vec::is_empty)
            }
        };
        if is_empty {
            bail_invalid_request!(param = "input", "input must not be empty.");
        }
        Ok(prompts)
    }
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct EmbeddingResponseData {
    pub index: usize,
    pub object: &'static str,
    pub embedding: EmbeddingData,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(super) enum EmbeddingData {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct EmbeddingResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub data: Vec<EmbeddingResponseData>,
    pub usage: PoolingUsage,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn parse(value: serde_json::Value) -> EmbeddingRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn request_variants_resolve_like_python() {
        let cases = [
            (json!({"input": "a"}), false, 1),
            (json!({"input": ["a", "b"]}), false, 2),
            (json!({"input": [1, 2]}), false, 1),
            (json!({"input": [[1], [2], [3]]}), false, 3),
            (
                json!({"messages": [{"role": "user", "content": "a"}]}),
                true,
                1,
            ),
            (
                json!({"messages": [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]}),
                true,
                2,
            ),
            (
                json!({"input": [{"role": "user", "content": "a"}]}),
                true,
                1,
            ),
            (
                json!({"input": [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]}),
                true,
                2,
            ),
        ];
        for (body, is_chat, count) in cases {
            let mut request = parse(body.clone());
            assert_eq!(request.is_chat(), is_chat, "{body}");
            let resolved = match request.take_prompts().unwrap() {
                EmbeddingPrompts::Completion(prompts) => prompts.len(),
                EmbeddingPrompts::Chat(conversations) => conversations.len(),
            };
            assert_eq!(resolved, count, "{body}");
        }
    }

    #[test]
    fn removed_and_invalid_parameters_are_rejected() {
        for (body, message) in [
            (
                json!({"input": "a", "normalize": true}),
                "Parameter `normalize` was removed; use `use_activation` instead.",
            ),
            (
                json!({"input": "a", "task": "score"}),
                "`score` task was removed; use `classify` instead.",
            ),
            (
                json!({"input": "a", "add_generation_prompt": true, "continue_final_message": true}),
                "Cannot set both `continue_final_message` and `add_generation_prompt` to True.",
            ),
        ] {
            let error = parse(body).validate_compat().unwrap_err();
            assert_eq!(error.to_error_response().error.message, message);
        }
        assert!(
            parse(json!({"input": "a", "user": "u", "padding": "max_length"}))
                .validate_compat()
                .is_ok()
        );
    }
}
