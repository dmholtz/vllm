// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Chat-message inputs shared by the pooling endpoints.

use std::collections::HashMap;

use itertools::Itertools as _;
use serde_json::Value;
use vllm_chat::{ChatOptions, ChatRequest, ResolvedToolContext, SamplingParams};
use vllm_text::Prompt;
use vllm_text::output::TextDecodeOptions;

use crate::error::{ApiError, bail_invalid_request, chat_submit_error};
use crate::routes::openai::chat_completions::convert::{
    convert_message, normalize_generation_prompt_mode,
};
use crate::routes::openai::utils::types::ChatMessage;
use crate::state::AppState;

/// Python `ChatRequestOptionsMixin` fields that affect template rendering.
#[derive(Debug, Clone, Default)]
pub(crate) struct ChatRenderOptions {
    pub add_generation_prompt: bool,
    pub continue_final_message: bool,
    pub chat_template: Option<String>,
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
}

/// Render chat conversations with the chat template into text prompts, like
/// Python's pooling IO processors (`EncodeChatRenderParams`).
pub(crate) async fn render_conversations(
    state: &AppState,
    options: &ChatRenderOptions,
    response_id: &str,
    conversations: Vec<Vec<ChatMessage>>,
) -> Result<Vec<Prompt>, ApiError> {
    let mut prompts = Vec::with_capacity(conversations.len());
    for (index, conversation) in conversations.into_iter().enumerate() {
        let messages: Vec<_> = conversation.into_iter().map(convert_message).try_collect()?;
        let generation_prompt_mode = normalize_generation_prompt_mode(
            Some(options.add_generation_prompt),
            options.continue_final_message,
            &messages,
        )?;
        let tool_context = ResolvedToolContext::new(&messages, Vec::new(), None, true)
            .map_err(|error| chat_submit_error("failed to resolve request tools", error))?;
        let request = ChatRequest {
            request_id: format!("{response_id}-{index}"),
            messages,
            sampling_params: SamplingParams::default(),
            chat_options: ChatOptions {
                generation_prompt_mode,
                chat_template: options.chat_template.clone(),
                reasoning_effort: None,
                response_format: None,
                template_kwargs: options.chat_template_kwargs.clone().unwrap_or_default(),
            },
            tool_context,
            decode_options: TextDecodeOptions::default(),
            intermediate: false,
            prompt_truncation: None,
            priority: 0,
            documents: None,
            cache_salt: None,
            add_special_tokens: false,
            data_parallel_rank: None,
            session_id: None,
            lora_request: None,
        };
        let rendered = state
            .chat
            .request_processor()
            .prepare_for_tokenization(request)
            .await
            .map_err(|error| chat_submit_error("failed to render chat input", error))?;
        // TODO: support multimodal pooling inputs.
        if rendered.mm_features.is_some() {
            bail_invalid_request!(
                param = "messages",
                "Multimodal embedding inputs are not supported by the Rust frontend yet."
            );
        }
        prompts.push(rendered.prompt);
    }
    Ok(prompts)
}
