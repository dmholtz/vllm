// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! RoBERTa byte-level BPE tokenizers built from `vocab.json` and `merges.txt`,
//! for checkpoints that ship no `tokenizer.json` (e.g.
//! `Jean-Baptiste/roberta-large-ner-english`).
//!
//! Port of transformers' `RobertaTokenizer.__init__` (transformers v5), which
//! the Python frontend runs when it loads such a checkpoint with
//! `AutoTokenizer`.

use std::path::Path;

use thiserror_ext::AsReport as _;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::roberta::RobertaProcessing;
use tokenizers::{AddedToken, Model as _, Tokenizer as HfTokenizer};

use crate::Result;

/// A named special token and the `AddedToken` flags that change how it is
/// matched in the input text. `normalized` is left out: RoBERTa has no
/// normalizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobertaSpecialToken {
    pub content: String,
    pub lstrip: bool,
    pub rstrip: bool,
    pub single_word: bool,
}

impl RobertaSpecialToken {
    /// A token without flags, as transformers creates it from a plain string.
    pub fn plain(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            lstrip: false,
            rstrip: false,
            single_word: false,
        }
    }
}

/// Settings of a RoBERTa tokenizer, as `RobertaTokenizer.__init__` takes them
/// from `tokenizer_config.json` and `special_tokens_map.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobertaBpeOptions {
    pub add_prefix_space: bool,
    pub trim_offsets: bool,
    pub bos_token: RobertaSpecialToken,
    pub eos_token: RobertaSpecialToken,
    pub unk_token: RobertaSpecialToken,
    pub sep_token: RobertaSpecialToken,
    pub pad_token: RobertaSpecialToken,
    pub cls_token: RobertaSpecialToken,
    pub mask_token: RobertaSpecialToken,
}

impl Default for RobertaBpeOptions {
    /// The `RobertaTokenizer.__init__` defaults.
    fn default() -> Self {
        Self {
            add_prefix_space: false,
            trim_offsets: true,
            bos_token: RobertaSpecialToken::plain("<s>"),
            eos_token: RobertaSpecialToken::plain("</s>"),
            unk_token: RobertaSpecialToken::plain("<unk>"),
            sep_token: RobertaSpecialToken::plain("</s>"),
            pad_token: RobertaSpecialToken::plain("<pad>"),
            cls_token: RobertaSpecialToken::plain("<s>"),
            mask_token: RobertaSpecialToken::plain("<mask>"),
        }
    }
}

impl RobertaBpeOptions {
    /// The named special tokens in transformers' `SPECIAL_TOKENS_ATTRIBUTES`
    /// order.
    pub fn special_tokens(&self) -> [&RobertaSpecialToken; 7] {
        [
            &self.bos_token,
            &self.eos_token,
            &self.unk_token,
            &self.sep_token,
            &self.pad_token,
            &self.cls_token,
            &self.mask_token,
        ]
    }
}

/// Build the tokenizer `RobertaTokenizer.__init__` produces from `vocab` and
/// `merges`.
pub(super) fn roberta_bpe_tokenizer(
    vocab: &Path,
    merges: &Path,
    options: &RobertaBpeOptions,
) -> Result<HfTokenizer> {
    fn utf8(path: &Path) -> Result<&str> {
        path.to_str()
            .ok_or_else(|| tokenizer_error!("path '{}' is not valid UTF-8", path.display()))
    }
    // No unk token, as in transformers: byte-level BPE covers every byte.
    let model = BPE::from_file(utf8(vocab)?, utf8(merges)?)
        .continuing_subword_prefix(String::new())
        .end_of_word_suffix(String::new())
        .fuse_unk(false)
        .build()
        .map_err(|error| {
            tokenizer_error!(
                "failed to load BPE vocab '{}' and merges '{}': {}",
                vocab.display(),
                merges.display(),
                error.as_report()
            )
        })?;
    let token_id = |token: &RobertaSpecialToken| {
        model.token_to_id(&token.content).ok_or_else(|| {
            tokenizer_error!(
                "special token '{}' is missing from BPE vocab '{}'",
                token.content,
                vocab.display()
            )
        })
    };
    let post_processor = RobertaProcessing::new(
        (
            options.sep_token.content.clone(),
            token_id(&options.sep_token)?,
        ),
        (
            options.cls_token.content.clone(),
            token_id(&options.cls_token)?,
        ),
    )
    .trim_offsets(options.trim_offsets)
    .add_prefix_space(options.add_prefix_space);

    let mut tokenizer = HfTokenizer::new(model);
    tokenizer
        .with_pre_tokenizer(Some(ByteLevel::new(options.add_prefix_space, true, true)))
        .with_post_processor(Some(post_processor))
        .with_decoder(Some(ByteLevelDecoder::default()));
    // transformers registers the named special tokens as special added tokens,
    // so `</s>` in the input text stays one token, and e.g. `<mask>` with
    // `lstrip` takes the space before it.
    let mut special_tokens: Vec<AddedToken> = Vec::new();
    for token in options.special_tokens() {
        if special_tokens.iter().all(|added| added.content != token.content) {
            special_tokens.push(
                AddedToken::from(token.content.clone(), true)
                    .lstrip(token.lstrip)
                    .rstrip(token.rstrip)
                    .single_word(token.single_word),
            );
        }
    }
    tokenizer.add_special_tokens(&special_tokens);
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::{RobertaBpeOptions, RobertaSpecialToken};
    use crate::{HuggingFaceTokenizer, Tokenizer as _};

    // A tiny byte-level BPE; the expected ids below are what transformers 5.16
    // `AutoTokenizer` produces for the same files.
    const VOCAB: &[&str] = &[
        "<s>", "<pad>", "</s>", "<unk>", "h", "e", "l", "o", "Ġ", "w", "r", "d", "!", "he", "ll",
        "hell", "hello", "Ġw", "or", "Ġwor", "Ġworl", "Ġworld", "Ġh", "Ġhe", "Ġhell", "Ġhello",
        "<mask>",
    ];
    const MERGES: &[&str] = &[
        "h e", "l l", "he ll", "hell o", "Ġ w", "o r", "Ġw or", "Ġwor l", "Ġworl d", "Ġ h", "Ġh e",
        "Ġhe ll", "Ġhell o",
    ];

    fn tokenizer(options: RobertaBpeOptions) -> HuggingFaceTokenizer {
        let dir = tempfile::tempdir().unwrap();
        let vocab = dir.path().join("vocab.json");
        let merges = dir.path().join("merges.txt");
        let map: serde_json::Map<_, _> = VOCAB
            .iter()
            .enumerate()
            .map(|(id, token)| (token.to_string(), id.into()))
            .collect();
        std::fs::write(&vocab, serde_json::to_string(&map).unwrap()).unwrap();
        std::fs::write(&merges, format!("#version: 0.2\n{}\n", MERGES.join("\n"))).unwrap();
        HuggingFaceTokenizer::from_roberta_vocab(&vocab, &merges, &options).unwrap()
    }

    #[test]
    fn defaults_match_transformers() {
        let t = tokenizer(RobertaBpeOptions::default());
        let cases: &[(&str, &[u32])] = &[
            ("hello world!", &[0, 16, 21, 12, 2]),
            (" hello world", &[0, 8, 16, 21, 2]),
            ("hello <mask>!", &[0, 16, 8, 26, 12, 2]),
            ("hello<mask>!", &[0, 16, 26, 12, 2]),
            ("hello </s> world", &[0, 16, 8, 2, 21, 2]),
            ("hex!", &[0, 13, 12, 2]),
            ("", &[0, 2]),
            ("world  hello", &[0, 9, 18, 6, 11, 8, 8, 16, 2]),
        ];
        for (text, ids) in cases {
            assert_eq!(&t.encode(text, true).unwrap(), ids, "{text:?}");
            assert_eq!(
                t.encode(text, false).unwrap(),
                ids[1..ids.len() - 1],
                "{text:?}"
            );
        }
        assert_eq!(t.decode(&[0, 16, 8, 26, 12, 2], true).unwrap(), "hello !");
        assert_eq!(
            t.decode(&[0, 16, 8, 26, 12, 2], false).unwrap(),
            "<s>hello <mask>!</s>"
        );
    }

    #[test]
    fn add_prefix_space_matches_transformers() {
        let t = tokenizer(RobertaBpeOptions {
            add_prefix_space: true,
            ..Default::default()
        });
        let cases: &[(&str, &[u32])] = &[
            ("hello world!", &[0, 8, 16, 21, 12, 2]),
            (" hello world", &[0, 8, 16, 21, 2]),
            ("hello <mask>!", &[0, 8, 16, 8, 26, 8, 12, 2]),
            ("hello </s> world", &[0, 8, 16, 8, 2, 21, 2]),
            ("world  hello", &[0, 21, 8, 8, 16, 2]),
        ];
        for (text, ids) in cases {
            assert_eq!(&t.encode(text, true).unwrap(), ids, "{text:?}");
        }
        assert_eq!(
            t.decode(&[0, 8, 16, 21, 12, 2], true).unwrap(),
            " hello world!"
        );
    }

    #[test]
    fn special_token_flags_and_names_match_transformers() {
        let mask = |lstrip, rstrip, single_word| RobertaSpecialToken {
            content: "<mask>".to_string(),
            lstrip,
            rstrip,
            single_word,
        };
        let t = tokenizer(RobertaBpeOptions {
            mask_token: mask(true, false, false),
            ..Default::default()
        });
        assert_eq!(t.encode("hello <mask>!", true).unwrap(), [0, 16, 26, 12, 2]);
        assert_eq!(
            t.decode(&[0, 16, 26, 12, 2], false).unwrap(),
            "<s>hello<mask>!</s>"
        );

        // single_word: `<mask>` inside a word is not the special token, and its
        // characters are not in the vocab (no unk token), so they are dropped.
        let t = tokenizer(RobertaBpeOptions {
            mask_token: mask(false, true, true),
            ..Default::default()
        });
        assert_eq!(t.encode("hello<mask>!", true).unwrap(), [0, 16, 12, 2]);
        assert_eq!(
            t.encode("hello <mask>!", true).unwrap(),
            [0, 16, 8, 26, 12, 2]
        );

        // cls and sep wrap the sequence, whatever their content.
        let t = tokenizer(RobertaBpeOptions {
            cls_token: RobertaSpecialToken::plain("<pad>"),
            sep_token: RobertaSpecialToken::plain("<unk>"),
            ..Default::default()
        });
        assert_eq!(t.encode("hello world!", true).unwrap(), [1, 16, 21, 12, 3]);
    }
}
