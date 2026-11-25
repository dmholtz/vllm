// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Selection and settings of the BERT WordPiece tokenizer for checkpoints that
//! ship only `vocab.txt` (no `tokenizer.json`), matching what transformers'
//! `AutoTokenizer` builds for them in the Python frontend.

use std::path::Path;

use serde::Deserialize;
use vllm_tokenizer::BertWordPieceOptions;

use super::config::NamedSpecialToken;
use crate::error::{Error, Result};

/// Tokenizer classes that transformers converts with `BertConverter`
/// (`SLOW_TO_FAST_CONVERTERS`), without the `Fast` suffix.
const BERT_TOKENIZER_CLASSES: &[&str] = &[
    "BertTokenizer",
    "DistilBertTokenizer",
    "DPRContextEncoderTokenizer",
    "DPRQuestionEncoderTokenizer",
    "DPRReaderTokenizer",
    "ElectraTokenizer",
    "LayoutLMTokenizer",
    "LxmertTokenizer",
    "MobileBertTokenizer",
    "RetriBertTokenizer",
    "SqueezeBertTokenizer",
];

/// `config.json` model types whose `AutoTokenizer` default is one of
/// [`BERT_TOKENIZER_CLASSES`] (`TOKENIZER_MAPPING_NAMES`), used when
/// `tokenizer_config.json` names no `tokenizer_class`.
const BERT_MODEL_TYPES: &[&str] = &[
    "bert",
    "distilbert",
    "electra",
    "layoutlm",
    "lxmert",
    "mobilebert",
    "squeezebert",
];

/// Whether transformers builds a BERT WordPiece tokenizer for this checkpoint.
pub(super) fn uses_bert_wordpiece(tokenizer_class: Option<&str>, model_type: Option<&str>) -> bool {
    match tokenizer_class {
        Some(class) => {
            BERT_TOKENIZER_CLASSES.contains(&class.strip_suffix("Fast").unwrap_or(class))
        }
        None => model_type.is_some_and(|model_type| BERT_MODEL_TYPES.contains(&model_type)),
    }
}

/// The `BertTokenizer.__init__` arguments of `tokenizer_config.json`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct BertTokenizerConfig {
    do_lower_case: Option<bool>,
    tokenize_chinese_chars: Option<bool>,
    strip_accents: Option<bool>,
    #[serde(flatten)]
    special_tokens: BertSpecialTokens,
}

/// The named special tokens of a BERT tokenizer.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct BertSpecialTokens {
    unk_token: Option<NamedSpecialToken>,
    sep_token: Option<NamedSpecialToken>,
    pad_token: Option<NamedSpecialToken>,
    cls_token: Option<NamedSpecialToken>,
    mask_token: Option<NamedSpecialToken>,
}

/// Read the tokenizer settings like `BertTokenizer.from_pretrained`:
/// `tokenizer_config.json` arguments over the `__init__` defaults, and
/// `special_tokens_map.json` entries over both.
///
/// `added_tokens.json` with entries is rejected: the Rust frontend does not
/// add those tokens yet.
// TODO: support non-empty `added_tokens.json`.
pub(super) fn bert_wordpiece_options(
    tokenizer_config: Option<&Path>,
    special_tokens_map: Option<&Path>,
    added_tokens: Option<&Path>,
) -> Result<BertWordPieceOptions> {
    if let Some(path) = added_tokens {
        let tokens: serde_json::Map<String, serde_json::Value> = read_json(path)?;
        if !tokens.is_empty() {
            return Err(Error::Tokenizer(format!(
                "'{}' adds {} tokens to a vocab.txt tokenizer, which is not supported",
                path.display(),
                tokens.len()
            )));
        }
    }
    let config: BertTokenizerConfig = match tokenizer_config {
        Some(path) => read_json(path)?,
        None => BertTokenizerConfig::default(),
    };
    let map: BertSpecialTokens = match special_tokens_map {
        Some(path) => read_json(path)?,
        None => BertSpecialTokens::default(),
    };

    let defaults = BertWordPieceOptions::default();
    let token =
        |from_map: Option<NamedSpecialToken>,
         from_config: Option<NamedSpecialToken>,
         default: String| { from_map.or(from_config).map(String::from).unwrap_or(default) };
    let special = config.special_tokens;
    Ok(BertWordPieceOptions {
        do_lower_case: config.do_lower_case.unwrap_or(defaults.do_lower_case),
        tokenize_chinese_chars: config
            .tokenize_chinese_chars
            .unwrap_or(defaults.tokenize_chinese_chars),
        strip_accents: config.strip_accents,
        unk_token: token(map.unk_token, special.unk_token, defaults.unk_token),
        sep_token: token(map.sep_token, special.sep_token, defaults.sep_token),
        pad_token: token(map.pad_token, special.pad_token, defaults.pad_token),
        cls_token: token(map.cls_token, special.cls_token, defaults.cls_token),
        mask_token: token(map.mask_token, special.mask_token, defaults.mask_token),
    })
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let bytes = std::fs::read(path).map_err(|error| {
        Error::Tokenizer(format!("failed to read '{}': {error}", path.display()))
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|error| Error::Tokenizer(format!("failed to parse '{}': {error}", path.display())))
}

#[cfg(test)]
mod tests {
    use thiserror_ext::AsReport as _;

    use super::*;

    #[test]
    fn bert_wordpiece_is_selected_by_class_or_model_type() {
        assert!(uses_bert_wordpiece(Some("BertTokenizerFast"), None));
        assert!(uses_bert_wordpiece(
            Some("DistilBertTokenizer"),
            Some("gpt2")
        ));
        assert!(uses_bert_wordpiece(None, Some("bert")));
        assert!(!uses_bert_wordpiece(Some("MPNetTokenizer"), Some("bert")));
        assert!(!uses_bert_wordpiece(None, Some("roberta")));
        assert!(!uses_bert_wordpiece(None, None));
    }

    #[test]
    fn options_follow_from_pretrained_precedence() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("tokenizer_config.json");
        let map = dir.path().join("special_tokens_map.json");
        let added = dir.path().join("added_tokens.json");
        std::fs::write(
            &config,
            r#"{"do_lower_case": false, "unk_token": "<cfg-unk>", "cls_token": {"content": "<cls>"}}"#,
        )
        .unwrap();
        std::fs::write(&map, r#"{"unk_token": "<map-unk>"}"#).unwrap();
        std::fs::write(&added, "{}").unwrap();

        let options = bert_wordpiece_options(Some(&config), Some(&map), Some(&added)).unwrap();
        expect_test::expect![[r#"
            BertWordPieceOptions {
                do_lower_case: false,
                tokenize_chinese_chars: true,
                strip_accents: None,
                unk_token: "<map-unk>",
                sep_token: "[SEP]",
                pad_token: "[PAD]",
                cls_token: "<cls>",
                mask_token: "[MASK]",
            }
        "#]]
        .assert_debug_eq(&options);

        std::fs::write(&added, r#"{"<new>": 30000}"#).unwrap();
        let error = bert_wordpiece_options(None, None, Some(&added)).unwrap_err();
        assert!(
            error.to_report_string().contains("adds 1 tokens"),
            "{error:?}"
        );
    }
}
