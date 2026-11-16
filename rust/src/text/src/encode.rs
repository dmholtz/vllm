// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use futures::future::try_join_all;
use vllm_engine_core_client::protocol::lora::LoraRequest;
use vllm_llm::{EncodeOutput, EncodeRequest, EngineTask, PoolingParams, PoolingTask};

use crate::{PoolingTokenization, Prompt, Result, TextLlm, TextRequestProcessor};

/// Text input and tokenization options for a pooling request.
#[derive(Debug, Clone)]
pub struct TextEncodeRequest {
    pub request_id: String,
    pub prompt: Prompt,
    pub task: PoolingTask,
    pub pooling_params: PoolingParams,
    /// Tokenization limits and options, like Python `TokenizeParams`.
    pub tokenization: PoolingTokenization,
    pub arrival_time: Option<f64>,
    pub cache_salt: Option<String>,
    pub trace_headers: Option<BTreeMap<String, String>>,
    pub priority: i32,
    pub data_parallel_rank: Option<u32>,
    pub session_id: Option<String>,
    pub lora_request: Option<LoraRequest>,
}

impl TextRequestProcessor {
    /// Tokenize and validate a pooling input without generation defaults.
    pub fn prepare_encode(&self, request: TextEncodeRequest) -> Result<EncodeRequest> {
        let prompt_token_ids =
            self.tokenize_pooling_prompt(request.prompt, &request.tokenization)?;
        self.validate_prompt_tokens(&request.request_id, &prompt_token_ids)?;
        Ok(EncodeRequest {
            request_id: request.request_id,
            prompt_token_ids,
            task: request.task,
            pooling_params: request.pooling_params,
            arrival_time: request.arrival_time,
            cache_salt: request.cache_salt,
            trace_headers: request.trace_headers,
            priority: request.priority,
            data_parallel_rank: request.data_parallel_rank,
            session_id: request.session_id,
            lora_request: request.lora_request,
        })
    }
}

impl TextLlm {
    /// Encode a batch in input order, validating all inputs before submission.
    ///
    /// Like Python's pooling serving, every input is submitted at once so the
    /// engine can batch them; the first failure aborts the remaining inputs.
    pub async fn encode_batch(
        &self,
        requests: Vec<TextEncodeRequest>,
    ) -> Result<Vec<EncodeOutput>> {
        let requests = requests
            .into_iter()
            .map(|request| self.processor.prepare_encode(request))
            .collect::<Result<Vec<_>>>()?;
        self.encode_prepared_batch(requests).await
    }

    /// Submit already-prepared token-level requests concurrently, returning
    /// outputs in input order.
    pub async fn encode_prepared_batch(
        &self,
        requests: Vec<EncodeRequest>,
    ) -> Result<Vec<EncodeOutput>> {
        Ok(try_join_all(requests.into_iter().map(|request| self.llm.encode(request))).await?)
    }

    /// Return engine-reported tasks, cached by the engine-core client.
    pub async fn supported_tasks(&self) -> Result<&[EngineTask]> {
        Ok(self.engine_core_client().get_supported_tasks().await?)
    }

    /// Resolve a classifier index through the model's label mapping.
    pub fn classification_label(&self, index: usize) -> Option<&str> {
        self.processor.backend.classification_label(index)
    }
}
