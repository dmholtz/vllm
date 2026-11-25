// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! BERT WordPiece tokenizers built from `vocab.txt`, for checkpoints that ship
//! no `tokenizer.json` (e.g. `dslim/bert-base-NER`).
//!
//! Port of transformers' `BertConverter` (`convert_slow_tokenizer.py`), which
//! the Python frontend runs when it loads such a checkpoint with
//! `AutoTokenizer`.

use std::path::Path;

use thiserror_ext::AsReport as _;
use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{AddedToken, Model as _, Tokenizer as HfTokenizer};

use crate::Result;

/// Settings of a BERT tokenizer, as `BertTokenizer.__init__` takes them from
/// `tokenizer_config.json` and `special_tokens_map.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BertWordPieceOptions {
    pub do_lower_case: bool,
    pub tokenize_chinese_chars: bool,
    /// `None` strips accents exactly when `do_lower_case` is set.
    pub strip_accents: Option<bool>,
    pub unk_token: String,
    pub sep_token: String,
    pub pad_token: String,
    pub cls_token: String,
    pub mask_token: String,
}

impl Default for BertWordPieceOptions {
    /// The `BertTokenizer.__init__` defaults.
    fn default() -> Self {
        Self {
            do_lower_case: true,
            tokenize_chinese_chars: true,
            strip_accents: None,
            unk_token: "[UNK]".to_string(),
            sep_token: "[SEP]".to_string(),
            pad_token: "[PAD]".to_string(),
            cls_token: "[CLS]".to_string(),
            mask_token: "[MASK]".to_string(),
        }
    }
}

/// Build the tokenizer `BertConverter.converted()` produces from `vocab`.
pub(super) fn bert_wordpiece_tokenizer(
    vocab: &Path,
    options: &BertWordPieceOptions,
) -> Result<HfTokenizer> {
    let vocab_str = vocab
        .to_str()
        .ok_or_else(|| tokenizer_error!("vocab path '{}' is not valid UTF-8", vocab.display()))?;
    let model = WordPiece::from_file(vocab_str)
        .unk_token(options.unk_token.clone())
        .build()
        .map_err(|error| {
            tokenizer_error!(
                "failed to load WordPiece vocab '{}': {}",
                vocab.display(),
                error.as_report()
            )
        })?;
    let token_id = |token: &str| {
        model.token_to_id(token).ok_or_else(|| {
            tokenizer_error!(
                "special token '{token}' is missing from WordPiece vocab '{}'",
                vocab.display()
            )
        })
    };
    let (cls, sep) = (options.cls_token.as_str(), options.sep_token.as_str());
    let (cls_id, sep_id) = (token_id(cls)?, token_id(sep)?);
    let template = |error: String| tokenizer_error!("invalid BERT template: {error}");
    let post_processor = TemplateProcessing::builder()
        .try_single(format!("{cls}:0 $A:0 {sep}:0"))
        .map_err(template)?
        .try_pair(format!("{cls}:0 $A:0 {sep}:0 $B:1 {sep}:1"))
        .map_err(template)?
        .special_tokens(vec![(cls, cls_id), (sep, sep_id)])
        .build()
        .map_err(|error| template(error.to_report_string()))?;

    let mut tokenizer = HfTokenizer::new(model);
    tokenizer
        .with_normalizer(Some(BertNormalizer::new(
            true,
            options.tokenize_chinese_chars,
            options.strip_accents,
            options.do_lower_case,
        )))
        .with_pre_tokenizer(Some(BertPreTokenizer))
        .with_post_processor(Some(post_processor))
        .with_decoder(Some(WordPieceDecoder::default()));
    // transformers registers the named special tokens as special added tokens,
    // so `[SEP]` in the input text stays one token instead of `[`, `SEP`, `]`.
    let special_tokens: Vec<AddedToken> = [
        &options.pad_token,
        &options.unk_token,
        &options.cls_token,
        &options.sep_token,
        &options.mask_token,
    ]
    .into_iter()
    .map(|token| AddedToken::from(token.clone(), true))
    .collect();
    tokenizer.add_special_tokens(&special_tokens);
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::BertWordPieceOptions;
    use crate::{HuggingFaceTokenizer, Tokenizer as _};

    const VOCAB: &[&str] = &[
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "Hello", "world", "!", "un",
        "##aff", "##able", "cafe", "café",
    ];

    fn tokenizer(options: BertWordPieceOptions) -> HuggingFaceTokenizer {
        let dir = tempfile::tempdir().unwrap();
        let vocab = dir.path().join("vocab.txt");
        std::fs::write(&vocab, VOCAB.join("\n")).unwrap();
        HuggingFaceTokenizer::from_bert_vocab(&vocab, &options).unwrap()
    }

    #[test]
    fn cased_vocab_keeps_case_and_wraps_in_cls_sep() {
        let t = tokenizer(BertWordPieceOptions {
            do_lower_case: false,
            ..Default::default()
        });
        // [CLS] Hello world ! [SEP]; unknown words map to [UNK]; WordPiece splits subwords.
        assert_eq!(t.encode("Hello world!", true).unwrap(), [2, 6, 7, 8, 3]);
        assert_eq!(
            t.encode("hello unaffable xyz", false).unwrap(),
            [5, 9, 10, 11, 1]
        );
        // Special tokens in the text stay single tokens, as in transformers.
        assert_eq!(t.encode("hello [SEP] world", false).unwrap(), [5, 3, 7]);
        assert_eq!(t.decode(&[2, 9, 10, 11, 3], true).unwrap(), "unaffable");
        // Accents are kept unless lowercasing (strip_accents: None).
        assert_eq!(t.encode("café", false).unwrap(), [13]);
    }

    #[test]
    fn uncased_vocab_lowercases_and_strips_accents() {
        let t = tokenizer(BertWordPieceOptions::default());
        assert_eq!(t.encode("Hello CAFÉ", false).unwrap(), [5, 12]);
    }
}
