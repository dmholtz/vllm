// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Selection and settings of the RoBERTa byte-level BPE tokenizer for
//! checkpoints that ship only `vocab.json` and `merges.txt` (no
//! `tokenizer.json`), matching what transformers' `AutoTokenizer` builds for
//! them in the Python frontend.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use vllm_tokenizer::{RobertaBpeOptions, RobertaSpecialToken};

use crate::error::{Error, Result};

/// Tokenizer classes that transformers resolves to `RobertaTokenizer`, without
/// the `Fast` suffix.
const ROBERTA_TOKENIZER_CLASSES: &[&str] = &[
    "RobertaTokenizer",
    "BartTokenizer",
    "LEDTokenizer",
    "LongformerTokenizer",
    "MvpTokenizer",
];

/// `config.json` model types whose `AutoTokenizer` default is
/// `RobertaTokenizer` (`TOKENIZER_MAPPING_NAMES`), used when
/// `tokenizer_config.json` names no `tokenizer_class`.
const ROBERTA_MODEL_TYPES: &[&str] = &[
    "bart",
    "bridgetower",
    "clap",
    "data2vec-text",
    "florence2",
    "ibert",
    "led",
    "lilt",
    "longformer",
    "mega",
    "mra",
    "mvp",
    "roberta",
    "roberta-prelayernorm",
];

/// Whether transformers builds a RoBERTa byte-level BPE tokenizer for this
/// checkpoint.
pub(super) fn uses_roberta_bpe(tokenizer_class: Option<&str>, model_type: Option<&str>) -> bool {
    match tokenizer_class {
        Some(class) => {
            ROBERTA_TOKENIZER_CLASSES.contains(&class.strip_suffix("Fast").unwrap_or(class))
        }
        None => model_type.is_some_and(|model_type| ROBERTA_MODEL_TYPES.contains(&model_type)),
    }
}

/// A special token entry: a plain string, or a serialized `AddedToken`.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum TokenEntry {
    Text(String),
    AddedToken(AddedTokenFlags),
}

#[derive(Debug, Clone, Deserialize)]
struct AddedTokenFlags {
    content: String,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    single_word: bool,
}

impl From<TokenEntry> for RobertaSpecialToken {
    fn from(entry: TokenEntry) -> Self {
        match entry {
            TokenEntry::Text(content) => Self::plain(content),
            TokenEntry::AddedToken(flags) => flags.into(),
        }
    }
}

impl From<AddedTokenFlags> for RobertaSpecialToken {
    fn from(flags: AddedTokenFlags) -> Self {
        Self {
            content: flags.content,
            lstrip: flags.lstrip,
            rstrip: flags.rstrip,
            single_word: flags.single_word,
        }
    }
}

/// The named special tokens of a RoBERTa tokenizer.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RobertaSpecialTokens {
    bos_token: Option<TokenEntry>,
    eos_token: Option<TokenEntry>,
    unk_token: Option<TokenEntry>,
    sep_token: Option<TokenEntry>,
    pad_token: Option<TokenEntry>,
    cls_token: Option<TokenEntry>,
    mask_token: Option<TokenEntry>,
}

/// The `RobertaTokenizer.__init__` arguments of `tokenizer_config.json`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RobertaTokenizerConfig {
    add_prefix_space: Option<bool>,
    trim_offsets: Option<bool>,
    /// Added tokens by id, as transformers >= 4.34 saves them.
    added_tokens_decoder: BTreeMap<String, AddedTokenFlags>,
    #[serde(flatten)]
    special_tokens: RobertaSpecialTokens,
}

/// Read the tokenizer settings like `RobertaTokenizer.from_pretrained` in
/// transformers v5:
///
/// - `tokenizer_config.json` arguments over the `__init__` defaults;
/// - for each named special token, the `special_tokens_map.json` entry over
///   the `tokenizer_config.json` one, flags included (a plain string there
///   drops the flags of a config entry);
/// - an `added_tokens_decoder` entry with the same content decides the flags
///   whenever it exists.
///
/// Added tokens other than the named special tokens (in
/// `added_tokens_decoder` or `added_tokens.json`) are rejected: the Rust
/// frontend does not add those yet.
// TODO: support added tokens that are not named special tokens.
pub(super) fn roberta_bpe_options(
    tokenizer_config: Option<&Path>,
    special_tokens_map: Option<&Path>,
    added_tokens: Option<&Path>,
) -> Result<RobertaBpeOptions> {
    if let Some(path) = added_tokens {
        let tokens: serde_json::Map<String, serde_json::Value> = read_json(path)?;
        if !tokens.is_empty() {
            return Err(Error::Tokenizer(format!(
                "'{}' adds {} tokens to a vocab.json/merges.txt tokenizer, which is not supported",
                path.display(),
                tokens.len()
            )));
        }
    }
    let config: RobertaTokenizerConfig = match tokenizer_config {
        Some(path) => read_json(path)?,
        None => RobertaTokenizerConfig::default(),
    };
    let map: RobertaSpecialTokens = match special_tokens_map {
        Some(path) => read_json(path)?,
        None => RobertaSpecialTokens::default(),
    };

    let decoder: Vec<AddedTokenFlags> = config.added_tokens_decoder.into_values().collect();
    let token = |from_map: Option<TokenEntry>,
                 from_config: Option<TokenEntry>,
                 default: RobertaSpecialToken| {
        let token = from_map.or(from_config).map(RobertaSpecialToken::from).unwrap_or(default);
        match decoder.iter().find(|added| added.content == token.content) {
            Some(added) => added.clone().into(),
            None => token,
        }
    };
    let defaults = RobertaBpeOptions::default();
    let special = config.special_tokens;
    let options = RobertaBpeOptions {
        add_prefix_space: config.add_prefix_space.unwrap_or(defaults.add_prefix_space),
        trim_offsets: config.trim_offsets.unwrap_or(defaults.trim_offsets),
        bos_token: token(map.bos_token, special.bos_token, defaults.bos_token),
        eos_token: token(map.eos_token, special.eos_token, defaults.eos_token),
        unk_token: token(map.unk_token, special.unk_token, defaults.unk_token),
        sep_token: token(map.sep_token, special.sep_token, defaults.sep_token),
        pad_token: token(map.pad_token, special.pad_token, defaults.pad_token),
        cls_token: token(map.cls_token, special.cls_token, defaults.cls_token),
        mask_token: token(map.mask_token, special.mask_token, defaults.mask_token),
    };

    let named = options.special_tokens().map(|token| token.content.as_str());
    let extra: Vec<&str> = decoder
        .iter()
        .map(|added| added.content.as_str())
        .filter(|content| !named.contains(content))
        .collect();
    if !extra.is_empty() {
        return Err(Error::Tokenizer(format!(
            "'{}' adds tokens that are not named special tokens to a vocab.json/merges.txt \
             tokenizer ({}), which is not supported",
            tokenizer_config.map_or_else(String::new, |path| path.display().to_string()),
            extra.join(", ")
        )));
    }
    Ok(options)
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
    fn roberta_bpe_is_selected_by_class_or_model_type() {
        assert!(uses_roberta_bpe(Some("RobertaTokenizerFast"), None));
        assert!(uses_roberta_bpe(Some("BartTokenizer"), Some("bert")));
        assert!(uses_roberta_bpe(None, Some("roberta")));
        assert!(uses_roberta_bpe(None, Some("longformer")));
        assert!(!uses_roberta_bpe(Some("GPT2Tokenizer"), Some("roberta")));
        assert!(!uses_roberta_bpe(None, Some("gpt2")));
        assert!(!uses_roberta_bpe(None, None));
    }

    struct Files {
        dir: tempfile::TempDir,
    }

    impl Files {
        fn new() -> Self {
            Self {
                dir: tempfile::tempdir().unwrap(),
            }
        }

        fn write(&self, name: &str, json: &str) -> std::path::PathBuf {
            let path = self.dir.path().join(name);
            std::fs::write(&path, json).unwrap();
            path
        }
    }

    fn mask(options: &RobertaBpeOptions) -> (bool, bool, bool) {
        let mask = &options.mask_token;
        (mask.lstrip, mask.rstrip, mask.single_word)
    }

    const MASK_LSTRIP: &str = r#"{"content": "<mask>", "lstrip": true, "rstrip": false, "single_word": false, "normalized": false}"#;

    #[test]
    fn options_follow_from_pretrained_precedence() {
        let files = Files::new();
        // The settings of Jean-Baptiste/roberta-large-ner-english.
        let config = files.write(
            "tokenizer_config.json",
            r#"{"unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>", "add_prefix_space": true,
                "errors": "replace", "sep_token": "</s>", "cls_token": "<s>", "pad_token": "<pad>",
                "mask_token": "<mask>", "model_max_length": 512, "name_or_path": "roberta-large"}"#,
        );
        let map = files.write(
            "special_tokens_map.json",
            &format!(
                r#"{{"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "sep_token": "</s>",
                    "pad_token": "<pad>", "cls_token": "<s>", "mask_token": {MASK_LSTRIP}}}"#
            ),
        );
        let added = files.write("added_tokens.json", "{}");
        let options = roberta_bpe_options(Some(&config), Some(&map), Some(&added)).unwrap();
        assert!(options.add_prefix_space);
        assert!(options.trim_offsets);
        assert_eq!(mask(&options), (true, false, false));
        assert_eq!(options.cls_token, RobertaSpecialToken::plain("<s>"));

        assert_eq!(
            roberta_bpe_options(None, None, None).unwrap(),
            RobertaBpeOptions::default()
        );
    }

    #[test]
    fn special_token_flags_follow_transformers() {
        let files = Files::new();
        let run = |config: &str, map: Option<&str>| {
            let config = files.write("tokenizer_config.json", config);
            let map = map.map(|map| files.write("special_tokens_map.json", map));
            mask(&roberta_bpe_options(Some(&config), map.as_deref(), None).unwrap())
        };
        let decoder = |entry: &str| format!(r#"{{"added_tokens_decoder": {{"26": {entry}}}}}"#);
        let mask_plain = MASK_LSTRIP.replace(r#""lstrip": true"#, r#""lstrip": false"#);

        // Flags of a config entry count, unless the map names the token plainly.
        let config_lstrip = format!(r#"{{"mask_token": {MASK_LSTRIP}}}"#);
        assert_eq!(run(&config_lstrip, None), (true, false, false));
        assert_eq!(
            run(&config_lstrip, Some(r#"{"mask_token": "<mask>"}"#)),
            (false, false, false)
        );
        // An added_tokens_decoder entry decides the flags.
        assert_eq!(
            run(&decoder(MASK_LSTRIP), Some(r#"{"mask_token": "<mask>"}"#)),
            (true, false, false)
        );
        assert_eq!(
            run(
                &decoder(&mask_plain),
                Some(&format!(r#"{{"mask_token": {MASK_LSTRIP}}}"#))
            ),
            (false, false, false)
        );
    }

    #[test]
    fn tokens_beyond_the_named_special_tokens_are_rejected() {
        let files = Files::new();
        let config = files.write(
            "tokenizer_config.json",
            r#"{"added_tokens_decoder": {"0": {"content": "<s>", "special": true},
                                         "27": {"content": "<new>", "special": false}}}"#,
        );
        let error = roberta_bpe_options(Some(&config), None, None).unwrap_err();
        assert!(error.to_report_string().contains("(<new>)"), "{error:?}");

        let added = files.write("added_tokens.json", r#"{"<new>": 27}"#);
        let error = roberta_bpe_options(None, None, Some(&added)).unwrap_err();
        assert!(
            error.to_report_string().contains("adds 1 tokens"),
            "{error:?}"
        );
    }
}
