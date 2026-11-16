// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;
use serde_json::Value;
use validator::Validate;
use vllm_llm::PoolingTask;
use vllm_text::{Prompt, TruncationSide};

use crate::error::{ApiError, bail_invalid_request, invalid_request};
use crate::routes::openai::utils::types::{ChatMessage, Normalizable};
use crate::routes::pooling_chat::ChatRenderOptions;
use crate::routes::pooling_encoding::{EmbedDType, EncodingFormat, Endianness};

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum Input {
    Text(String),
    TextBatch(Vec<String>),
    TokenIds(Vec<u32>),
    TokenIdBatch(Vec<Vec<u32>>),
}

impl Input {
    pub(super) fn into_prompts(self) -> Vec<Prompt> {
        match self {
            Self::Text(text) => vec![Prompt::Text(text)],
            Self::TextBatch(batch) => batch.into_iter().map(Prompt::Text).collect(),
            Self::TokenIds(token_ids) => vec![Prompt::TokenIds(token_ids)],
            Self::TokenIdBatch(batch) => batch.into_iter().map(Prompt::TokenIds).collect(),
        }
    }
}

/// Python `padding` tokenization option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Padding {
    MaxLength,
    DoNotPad,
}

/// `/pooling` and `/classify` request.
///
/// Covers Python's `PoolingCompletionRequest` / `PoolingChatRequest` and
/// `ClassificationCompletionRequest` / `ClassificationChatRequest`.
#[derive(Debug, Clone, Deserialize, Validate)]
pub(crate) struct PoolingRequest {
    pub model: Option<String>,
    /// Completion inputs.
    pub input: Option<Input>,
    /// One chat conversation to render with the chat template.
    pub messages: Option<Vec<ChatMessage>>,
    /// Requested pooling task, as its wire literal.
    pub task: Option<String>,
    pub dimensions: Option<u32>,
    pub use_activation: Option<bool>,
    #[serde(default)]
    pub encoding_format: EncodingFormat,
    #[serde(default)]
    pub embed_dtype: EmbedDType,
    #[serde(default)]
    pub endianness: Endianness,
    /// Defaults to `true` for completion inputs and `false` for chat inputs.
    pub add_special_tokens: Option<bool>,
    pub truncate_prompt_tokens: Option<i64>,
    pub truncation_side: Option<TruncationSide>,
    pub padding: Option<Padding>,
    pub request_id: Option<String>,
    #[serde(default)]
    pub priority: i32,
    pub cache_salt: Option<String>,
    #[serde(default)]
    pub add_generation_prompt: bool,
    #[serde(default)]
    pub continue_final_message: bool,
    pub chat_template: Option<String>,
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    pub mm_processor_kwargs: Option<Value>,
    pub media_io_kwargs: Option<Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Normalizable for PoolingRequest {}

impl PoolingRequest {
    pub(super) fn validate_options(&self) -> Result<(), ApiError> {
        // `reject_removed_pooling_parameters`.
        if self.extra.contains_key("normalize") {
            bail_invalid_request!(
                param = "normalize",
                "Parameter `normalize` was removed; use `use_activation` instead."
            );
        }
        // TODO: integrate IOProcessor plugins (`IOProcessorRequest.data`).
        if self.extra.contains_key("data") {
            bail_invalid_request!(
                param = "data",
                "IOProcessor plugin requests are not supported by the Rust frontend yet."
            );
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
        match (&self.input, &self.messages) {
            (None, None) => bail_invalid_request!(
                param = "input",
                "Either `input` or `messages` must be provided."
            ),
            (_, Some(messages)) if messages.is_empty() => {
                bail_invalid_request!(param = "messages", "messages must not be empty.")
            }
            _ => Ok(()),
        }
    }

    /// Parse the requested task, rejecting removed task names like Python's
    /// `check_removed_pooling_task`.
    pub(super) fn task(&self) -> Result<Option<PoolingTask>, ApiError> {
        use serde::de::IntoDeserializer as _;

        let Some(task) = self.task.as_deref() else {
            return Ok(None);
        };
        match task {
            "score" => bail_invalid_request!(
                param = "task",
                "`score` task was removed; use `classify` instead."
            ),
            "encode" => bail_invalid_request!(
                param = "task",
                "`encode` task was removed; use `token_embed` or `token_classify` instead."
            ),
            _ => {}
        }
        PoolingTask::deserialize(task.into_deserializer()).map(Some).map_err(
            |_: serde::de::value::Error| {
                invalid_request!(param = "task", "Invalid pooling task {task:?}.")
            },
        )
    }

    pub(super) fn chat_render_options(&self) -> ChatRenderOptions {
        ChatRenderOptions {
            add_generation_prompt: self.add_generation_prompt,
            continue_final_message: self.continue_final_message,
            chat_template: self.chat_template.clone(),
            chat_template_kwargs: self.chat_template_kwargs.clone(),
        }
    }
}
