// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use super::OpaqueValue;
use super::task::PoolingTask;

/// Python `RequestOutputKind.FINAL_ONLY`, the only output kind accepted for
/// pooling requests.
pub const OUTPUT_KIND_FINAL_ONLY: u8 = 2;

fn default_output_kind() -> u8 {
    OUTPUT_KIND_FINAL_ONLY
}

/// Mode of a worker-side late-interaction scoring request.
///
/// Original Python definition: `LateInteractionParams.mode` in
/// `vllm/pooling_params.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LateInteractionMode {
    /// Cache the query token embeddings on the worker.
    CacheQuery,
    /// Score a document against a previously cached query.
    ScoreDoc,
}

/// Metadata for worker-side late-interaction scoring.
///
/// Python encodes this `msgspec.Struct(array_like=True)` positionally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
pub struct LateInteractionParams {
    pub mode: LateInteractionMode,
    /// Stable key used for both DP routing and the worker cache lookup.
    pub query_key: String,
    /// Expected number of document requests for a `cache_query` request.
    #[serde(default)]
    pub query_uses: Option<u64>,
}

/// Fixed-length mean pooling before the token embedding head.
///
/// Python encodes this `msgspec.Struct(array_like=True)` positionally.
#[derive(Debug, Clone, PartialEq, Eq, Serialize_tuple, Deserialize_tuple)]
pub struct LateChunkingParams {
    /// Tokens per chunk.
    pub chunk_size: u64,
}

/// API parameters for pooling models.
///
/// Mirrors Python's array-like `PoolingParams` field by field, in declaration
/// order. msgspec writes every field of an `array_like` struct even with
/// `omit_defaults=True`, so Rust does the same; decoding tolerates shorter
/// arrays from older peers.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/pooling_params.py>
///
/// Original Python field documentation:
/// <https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/pooler.py>
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct EngineCorePoolingParams {
    /// Whether to apply activation function to the pooler outputs.
    /// `None` lets engine-core resolve the model's default.
    pub use_activation: Option<bool>,
    /// Reduce the dimensions of embeddings if model support matryoshka
    /// representation.
    /// `None` lets engine-core resolve the model's default.
    pub dimensions: Option<u32>,
    /// If set, only the score corresponding to the `step_tag_id` in the
    /// generated sentence should be returned. Otherwise, the scores for all
    /// tokens are returned.
    pub step_tag_id: Option<u32>,
    /// A list of indices for the vocabulary dimensions to be extracted,
    /// such as the token IDs of `good_token` and `bad_token` in the
    /// `math-shepherd-mistral-7b-prm` model.
    pub returned_token_ids: Option<Vec<u32>>,
    /// The task used for pooling.
    pub task: PoolingTask,
    /// Internal: whether the pooler needs the prompt token IDs.
    #[serde(default)]
    pub requires_token_ids: bool,
    /// Internal: skip reading the prefix cache. `None` lets engine-core
    /// resolve the task default.
    #[serde(default)]
    pub skip_reading_prefix_cache: Option<bool>,
    /// Internal: late-interaction scoring metadata.
    #[serde(default)]
    pub late_interaction_params: Option<LateInteractionParams>,
    /// Internal: model-specific keyword arguments, such as
    /// `compressed_token_type_ids` for cross-encoders.
    #[serde(default)]
    pub extra_kwargs: Option<BTreeMap<String, OpaqueValue>>,
    /// Python `RequestOutputKind`; pooling only supports `FINAL_ONLY`.
    #[serde(default = "default_output_kind")]
    pub output_kind: u8,
    /// Mean-pool contextual tokens into fixed-size chunks for `token_embed`.
    #[serde(default)]
    pub late_chunking_params: Option<LateChunkingParams>,
}

impl Default for EngineCorePoolingParams {
    fn default() -> Self {
        Self {
            use_activation: None,
            dimensions: None,
            step_tag_id: None,
            returned_token_ids: None,
            task: PoolingTask::default(),
            requires_token_ids: false,
            skip_reading_prefix_cache: None,
            late_interaction_params: None,
            extra_kwargs: None,
            output_kind: OUTPUT_KIND_FINAL_ONLY,
            late_chunking_params: None,
        }
    }
}

impl EngineCorePoolingParams {
    /// Return the late-interaction query key used for DP affinity routing, if
    /// any.
    pub fn late_interaction_query_key(&self) -> Option<&str> {
        self.late_interaction_params
            .as_ref()
            .map(|params| params.query_key.as_str())
            .filter(|key| !key.is_empty())
    }
}

/// Model-dependent pooling settings resolved by engine-core.
///
/// Returned by the `get_pooling_frontend_config` utility, so frontends that do
/// not share Python's `ModelConfig` see the same resolved pooler settings
/// (after `--pooler-config`, `--hf-overrides` and Sentence-Transformers
/// merging) as the Python pooling entrypoints.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PoolingFrontendConfig {
    /// Python `ModelConfig.get_pooling_task(supported_tasks)`: the task served
    /// by the task-specific endpoints.
    #[serde(deserialize_with = "deserialize_optional_task_literal")]
    pub pooling_task: Option<PoolingTask>,
    /// Split over-long embedding inputs into `max_model_len` chunks and
    /// aggregate the chunk embeddings.
    pub enable_chunked_processing: bool,
    /// Maximum embedding input length; `None` falls back to `max_model_len`
    /// unless chunked processing is enabled.
    pub max_embed_len: Option<u32>,
    /// Pooler-level default for `use_activation`; `None` means the task
    /// default (`true` for embedding and classification).
    pub use_activation: Option<bool>,
    /// Whether late-interaction scoring uses worker-side query caching.
    pub enable_flash_late_interaction: bool,
    /// Number of classifier labels, from the HF config.
    pub num_labels: Option<u32>,
    /// Classifier label names by class index (string keys, as in HF JSON).
    pub id2label: BTreeMap<String, String>,
    /// Whether cross-encoders separate the query and document with a SEP token.
    pub use_sep_token: bool,
    /// Resolved model architecture name.
    pub architecture: Option<String>,
    /// Lowercase text before tokenization (from `encoder_config`).
    pub do_lower_case: bool,
    /// Whether the model supports Matryoshka `dimensions`.
    pub is_matryoshka: bool,
    /// Explicitly supported Matryoshka dimensions, if restricted.
    pub matryoshka_dimensions: Option<Vec<u32>>,
    /// Size of the full embedding vector.
    pub embedding_size: Option<u32>,
    /// Whether the model accepts multimodal inputs.
    pub is_multimodal_model: bool,
}

/// Decode an optional task literal through its string form: utility results
/// are decoded with `rmpv::ext::from_value`, which cannot drive a unit-variant
/// enum from a msgpack string.
fn deserialize_optional_task_literal<'de, D>(
    deserializer: D,
) -> Result<Option<PoolingTask>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::IntoDeserializer as _;

    Option::<String>::deserialize(deserializer)?
        .map(|literal| {
            PoolingTask::deserialize(literal.as_str().into_deserializer())
                .map_err(|error: serde::de::value::Error| serde::de::Error::custom(error))
        })
        .transpose()
}

impl PoolingFrontendConfig {
    /// Resolve a classifier label by class index.
    pub fn label(&self, index: usize) -> Option<&str> {
        self.id2label.get(&index.to_string()).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{decode_msgpack, encode_msgpack};

    /// Fixture produced by Python v0.33.0:
    ///
    /// ```python
    /// msgspec.msgpack.encode(PoolingParams(
    ///     task="token_embed", use_activation=True, dimensions=8,
    ///     late_interaction_params=LateInteractionParams(
    ///         mode="cache_query", query_key="k", query_uses=2),
    ///     extra_kwargs={"compressed_token_type_ids": (3, 7)},
    ///     late_chunking_params=LateChunkingParams(chunk_size=4),
    /// )).hex()
    /// ```
    const PYTHON_FULL_FIXTURE: &str = "9bc308c0c0ab746f6b656e5f656d626564c2c093ab63616368655f7175657279a16b0281b9636f6d707265737365645f746f6b656e5f747970655f696473920307029104";

    /// `msgspec.msgpack.encode(PoolingParams(task="embed")).hex()`.
    const PYTHON_DEFAULT_EMBED_FIXTURE: &str = "9bc0c0c0c0a5656d626564c2c0c0c002c0";

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn full_params() -> EngineCorePoolingParams {
        EngineCorePoolingParams {
            use_activation: Some(true),
            dimensions: Some(8),
            task: PoolingTask::TokenEmbed,
            late_interaction_params: Some(LateInteractionParams {
                mode: LateInteractionMode::CacheQuery,
                query_key: "k".to_string(),
                query_uses: Some(2),
            }),
            extra_kwargs: Some(BTreeMap::from([(
                "compressed_token_type_ids".to_string(),
                OpaqueValue::Array(vec![3.into(), 7.into()]),
            )])),
            late_chunking_params: Some(LateChunkingParams { chunk_size: 4 }),
            ..Default::default()
        }
    }

    #[test]
    fn encoding_matches_python_byte_for_byte() {
        assert_eq!(
            hex(&encode_msgpack(&full_params()).unwrap()),
            PYTHON_FULL_FIXTURE
        );
        let embed = EngineCorePoolingParams {
            task: PoolingTask::Embed,
            ..Default::default()
        };
        assert_eq!(
            hex(&encode_msgpack(&embed).unwrap()),
            PYTHON_DEFAULT_EMBED_FIXTURE
        );
    }

    #[test]
    fn decoding_accepts_full_and_truncated_arrays() {
        let full = (0..PYTHON_FULL_FIXTURE.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&PYTHON_FULL_FIXTURE[i..i + 2], 16).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            decode_msgpack::<EngineCorePoolingParams>(&full).unwrap(),
            full_params()
        );

        // Older frontends sent only the five public fields.
        let prefix = encode_msgpack(&(
            Some(true),
            None::<u32>,
            None::<u32>,
            None::<Vec<u32>>,
            "embed",
        ))
        .unwrap();
        assert_eq!(
            decode_msgpack::<EngineCorePoolingParams>(&prefix).unwrap(),
            EngineCorePoolingParams {
                use_activation: Some(true),
                task: PoolingTask::Embed,
                ..Default::default()
            }
        );
    }

    #[test]
    fn frontend_config_decodes_python_dict_with_missing_fields() {
        let value = rmpv::Value::Map(vec![
            ("pooling_task".into(), "embed".into()),
            ("enable_chunked_processing".into(), true.into()),
            ("max_embed_len".into(), rmpv::Value::Nil),
            (
                "id2label".into(),
                rmpv::Value::Map(vec![("0".into(), "NEG".into()), ("1".into(), "POS".into())]),
            ),
            ("unknown_future_field".into(), 1.into()),
        ]);
        let config: PoolingFrontendConfig = rmpv::ext::from_value(value).unwrap();
        assert_eq!(config.pooling_task, Some(PoolingTask::Embed));
        assert!(config.enable_chunked_processing);
        assert_eq!(config.max_embed_len, None);
        assert_eq!(config.label(1), Some("POS"));
        assert_eq!(config.label(2), None);
    }

    #[test]
    fn late_interaction_query_key_ignores_empty_keys() {
        let mut params = full_params();
        assert_eq!(params.late_interaction_query_key(), Some("k"));
        params.late_interaction_params.as_mut().unwrap().query_key.clear();
        assert_eq!(params.late_interaction_query_key(), None);
    }
}
