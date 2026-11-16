// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use thiserror::Error;
use vllm_engine_core_client::protocol::output::RequestError;
use vllm_engine_core_client::protocol::task::EngineTask;

pub type Result<T> = std::result::Result<T, Error>;

/// Public error type for the Rust `llm` facade.
#[derive(Debug, Error)]
pub enum Error {
    #[error("request `{request_id}` has an empty prompt_token_ids")]
    EmptyPromptTokenIds { request_id: String },
    #[error("pooling request `{request_id}` failed: {message}")]
    PoolingRequest { request_id: String, message: String },
    #[error(
        "model `{model_name}` does not support task {requested_task:?}; supported tasks: {supported_tasks:?}"
    )]
    UnsupportedTask {
        model_name: String,
        requested_task: EngineTask,
        supported_tasks: Vec<EngineTask>,
    },
    #[error("request `{request_id}` was rejected by engine-core{}", error.as_ref().map(|e| format!(": {}", e.message)).unwrap_or_default())]
    EngineRequest {
        request_id: String,
        error: Option<Box<RequestError>>,
    },
    #[error("engine-core error")]
    EngineCoreClient(#[from] vllm_engine_core_client::Error),
    #[error(
        "sampling mask for request `{request_id}` has {row_count} rows for {token_count} generated tokens"
    )]
    SamplingMaskTokenCountMismatch {
        request_id: String,
        token_count: usize,
        row_count: usize,
    },
    #[error("routed experts for request `{request_id}` changed dtype or shape")]
    RoutedExpertsShapeMismatch { request_id: String },
}
