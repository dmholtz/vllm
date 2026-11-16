// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Engine-reported tasks that decide which HTTP routes this frontend serves.

use vllm_engine_core_client::protocol::pooling::PoolingFrontendConfig;
use vllm_llm::{EngineTask, GenerationTask, PoolingTask};

/// Tasks and resolved pooling settings discovered from engine-core at startup.
///
/// Mirrors the inputs of Python's `build_app()` route registration
/// (`vllm/entrypoints/launchers/api_server/routers.py`) and
/// `register_pooling_api_routers` (`vllm/entrypoints/pooling/factories.py`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ServingTasks {
    /// Tasks supported by every connected engine.
    pub supported: Vec<EngineTask>,
    /// Resolved pooling settings; `None` for models without a pooler.
    pub pooling: Option<PoolingFrontendConfig>,
}

impl ServingTasks {
    fn supports(&self, task: impl Into<EngineTask>) -> bool {
        self.supported.contains(&task.into())
    }

    /// Whether generation routes (`/v1/completions`, `/v1/chat/completions`,
    /// ...) are served.
    pub fn supports_generate(&self) -> bool {
        self.supports(GenerationTask::Generate)
    }

    /// Python `ModelConfig.get_pooling_task(supported_tasks)`.
    pub fn pooling_task(&self) -> Option<PoolingTask> {
        self.pooling.as_ref().and_then(|config| config.pooling_task)
    }

    /// Whether `/pooling` is served.
    pub fn serves_pooling(&self) -> bool {
        self.pooling_task().is_some()
    }

    /// Whether `/classify` is served.
    pub fn serves_classify(&self) -> bool {
        self.supports(PoolingTask::Classify)
    }

    /// Whether `/v1/embeddings` is served by the sequence embedding handler.
    pub fn serves_embeddings(&self) -> bool {
        self.pooling_task() != Some(PoolingTask::TokenEmbed) && self.supports(PoolingTask::Embed)
    }

    /// Whether `/v1/embeddings` is served by the token embedding handler.
    pub fn serves_token_embeddings(&self) -> bool {
        self.pooling_task() == Some(PoolingTask::TokenEmbed)
    }

    /// Python `enable_scoring_api`: embedding models score with a
    /// bi-encoder or late interaction, and single-label classifiers with a
    /// cross-encoder.
    pub fn serves_scoring(&self) -> bool {
        match self.pooling_task() {
            Some(PoolingTask::Embed | PoolingTask::TokenEmbed) => true,
            Some(PoolingTask::Classify) => {
                self.pooling.as_ref().and_then(|config| config.num_labels) == Some(1)
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tasks(supported: &[EngineTask], pooling_task: Option<PoolingTask>) -> ServingTasks {
        ServingTasks {
            supported: supported.to_vec(),
            pooling: pooling_task.map(|task| PoolingFrontendConfig {
                pooling_task: Some(task),
                ..Default::default()
            }),
        }
    }

    #[test]
    fn route_gating_matches_python_registration() {
        use EngineTask::{Generation, Pooling};

        let generate = tasks(&[Generation(GenerationTask::Generate)], None);
        assert!(generate.supports_generate());
        assert!(!generate.serves_pooling() && !generate.serves_embeddings());

        let embed = tasks(
            &[
                Pooling(PoolingTask::Embed),
                Pooling(PoolingTask::TokenEmbed),
            ],
            Some(PoolingTask::Embed),
        );
        assert!(!embed.supports_generate());
        assert!(embed.serves_pooling() && embed.serves_embeddings() && embed.serves_scoring());
        assert!(!embed.serves_classify() && !embed.serves_token_embeddings());

        let token_embed = tasks(
            &[Pooling(PoolingTask::TokenEmbed)],
            Some(PoolingTask::TokenEmbed),
        );
        assert!(token_embed.serves_token_embeddings() && !token_embed.serves_embeddings());
        assert!(token_embed.serves_scoring());

        let mut classify = tasks(
            &[Pooling(PoolingTask::Classify)],
            Some(PoolingTask::Classify),
        );
        assert!(classify.serves_classify() && !classify.serves_scoring());
        classify.pooling.as_mut().unwrap().num_labels = Some(1);
        assert!(classify.serves_scoring());
    }
}
