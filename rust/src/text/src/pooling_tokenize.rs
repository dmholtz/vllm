// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Prompt tokenization for pooling requests.
//!
//! Mirrors the subset of Python `vllm.renderers.params.TokenizeParams` used by
//! the pooling entrypoints (`_build_pooling_tok_params`). Pooling requests
//! generate no output tokens, so `truncate_prompt_tokens=-1` keeps the whole
//! input budget instead of holding one token back as generation does.

use vllm_tokenizer::Tokenizer;

use crate::error::{Error, Result};
use crate::{Prompt, TextRequestProcessor, TruncationSide};

/// Tokenization limits and options for one pooling prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolingTokenization {
    /// Maximum prompt length (`max_total_tokens`); `None` disables the length
    /// check and makes `truncate_prompt_tokens=-1` a no-op.
    pub max_total_tokens: Option<u32>,
    /// Number of tokens to keep. `-1` keeps `max_total_tokens`; `None` leaves
    /// the prompt untruncated.
    pub truncate_prompt_tokens: Option<i64>,
    /// Side to truncate from. `None` lets the tokenizer truncate text
    /// prompts, which keeps template special tokens such as a trailing
    /// `[SEP]`, and keeps the leading tokens of token-ID prompts.
    pub truncation_side: Option<TruncationSide>,
    /// Pad the prompt with the tokenizer's pad token up to this many tokens.
    pub pad_prompt_tokens: Option<u32>,
    /// Lowercase text prompts before tokenization.
    pub do_lower_case: bool,
    /// Whether tokenization adds special tokens such as BOS/CLS.
    pub add_special_tokens: bool,
}

impl PoolingTokenization {
    /// Python `TokenizeParams` for pooling: limit prompts to `max_model_len`
    /// without padding.
    pub fn new(max_total_tokens: Option<u32>, add_special_tokens: bool) -> Self {
        Self {
            max_total_tokens,
            truncate_prompt_tokens: None,
            truncation_side: None,
            pad_prompt_tokens: None,
            do_lower_case: false,
            add_special_tokens,
        }
    }

    /// Python `TokenizeParams.__post_init__` checks.
    fn validate(&self) -> Result<()> {
        match self.truncate_prompt_tokens {
            Some(value) if value < -1 => Err(Error::InvalidTruncatePromptTokens { value }),
            Some(value) if value >= 0 => match self.max_total_tokens {
                Some(max_total_tokens) if value > i64::from(max_total_tokens) => {
                    Err(Error::TruncatePromptTokensExceedsMaxModelLen {
                        value: value as u64,
                        max_model_len: max_total_tokens,
                    })
                }
                _ => Ok(()),
            },
            _ => Ok(()),
        }
    }

    /// Resolved truncation length; `-1` maps to the full input budget.
    fn truncation_limit(&self) -> Option<usize> {
        match self.truncate_prompt_tokens? {
            -1 => self.max_total_tokens.map(|max| max as usize),
            value => Some(value as usize),
        }
    }

    /// Length the tokenizer truncates text prompts to, like Python
    /// `TokenizeParams.get_encode_kwargs`. An explicit side disables tokenizer
    /// truncation; without explicit truncation the prompt is capped one token
    /// past the limit, so over-long prompts still fail the length check.
    fn tokenizer_truncation_limit(&self) -> Option<usize> {
        if self.truncation_side.is_some() && self.truncate_prompt_tokens.is_some() {
            return None;
        }
        self.truncation_limit()
            .or_else(|| self.max_total_tokens.map(|max| max as usize + 1))
    }
}

/// Encode `text` truncated to `max_length` tokens the way a Hugging Face
/// tokenizer does with `truncation=True`: the content is truncated from the
/// right and the template special tokens (e.g. `[CLS]`/`[SEP]`) are kept.
fn encode_truncated_like_hf(
    tokenizer: &dyn Tokenizer,
    text: &str,
    add_special_tokens: bool,
    max_length: usize,
) -> Result<Vec<u32>> {
    let full = tokenizer.encode(text, add_special_tokens)?;
    if full.len() <= max_length {
        return Ok(full);
    }
    if !add_special_tokens {
        return Ok(full[..max_length].to_vec());
    }

    // `full` is `prefix + content + suffix`; locate the content to keep the
    // special tokens around it.
    let content = tokenizer.encode(text, false)?;
    let Some(num_special) = full.len().checked_sub(content.len()) else {
        return Ok(full[..max_length].to_vec());
    };
    let Some(prefix_len) =
        (0..=num_special).find(|&prefix| full[prefix..prefix + content.len()] == content[..])
    else {
        return Ok(full[..max_length].to_vec());
    };
    let keep = max_length.saturating_sub(num_special).min(content.len());
    let mut truncated = Vec::with_capacity(num_special + keep);
    truncated.extend_from_slice(&full[..prefix_len]);
    truncated.extend_from_slice(&content[..keep]);
    truncated.extend_from_slice(&full[prefix_len + content.len()..]);
    Ok(truncated)
}

impl TextRequestProcessor {
    /// Tokenize and validate one pooling prompt like Python's
    /// `TokenizeParams.apply_pre_tokenization` / `apply_post_tokenization`:
    /// lowercase, tokenize, truncate, pad, then check the length.
    pub fn tokenize_pooling_prompt(
        &self,
        prompt: Prompt,
        params: &PoolingTokenization,
    ) -> Result<Vec<u32>> {
        params.validate()?;
        let mut tokens = match prompt {
            Prompt::Text(text) => {
                let text = if params.do_lower_case {
                    text.to_lowercase()
                } else {
                    text
                };
                let tokenizer = self.backend.tokenizer();
                match params.tokenizer_truncation_limit() {
                    Some(limit) => encode_truncated_like_hf(
                        tokenizer.as_ref(),
                        &text,
                        params.add_special_tokens,
                        limit,
                    )?,
                    None => tokenizer.encode(&text, params.add_special_tokens)?,
                }
            }
            Prompt::TokenIds(token_ids) => token_ids,
        };

        if let Some(limit) = params.truncation_limit()
            && limit < tokens.len()
        {
            match params.truncation_side.unwrap_or(TruncationSide::Right) {
                TruncationSide::Left => {
                    tokens.drain(..tokens.len() - limit);
                }
                TruncationSide::Right => tokens.truncate(limit),
            }
        }

        if let Some(pad_length) = params.pad_prompt_tokens.map(|length| length as usize)
            && pad_length > tokens.len()
        {
            let pad_token_id = self.backend.pad_token_id().ok_or(Error::MissingPadToken)?;
            tokens.resize(pad_length, pad_token_id);
        }

        if let Some(max_total_tokens) = params.max_total_tokens
            && tokens.len() > max_total_tokens as usize
        {
            return Err(Error::PoolingPromptTooLong {
                max_total_tokens,
                prompt_len: tokens.len(),
                // The tokenizer may have capped the prompt one token past the
                // limit, so the real prompt may be longer.
                qualifier: if tokens.len() == max_total_tokens as usize + 1 {
                    "at least "
                } else {
                    ""
                },
            });
        }

        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use thiserror_ext::AsReport as _;
    use vllm_tokenizer::DynTokenizer;
    use vllm_tokenizer::test_utils::TestTokenizer;

    use super::*;
    use crate::TextBackend;

    /// Byte-level test tokenizer with BERT-like `[CLS] ... [SEP]` specials.
    struct BertLikeTokenizer(TestTokenizer);

    impl Tokenizer for BertLikeTokenizer {
        fn encode(&self, text: &str, add_special_tokens: bool) -> vllm_tokenizer::Result<Vec<u32>> {
            let mut ids = self.0.encode(text, false)?;
            if add_special_tokens {
                ids.insert(0, 301);
                ids.push(302);
            }
            Ok(ids)
        }

        fn encode_ordinary(&self, text: &str) -> vllm_tokenizer::Result<Vec<u32>> {
            self.0.encode_ordinary(text)
        }

        fn decode(&self, ids: &[u32], skip: bool) -> vllm_tokenizer::Result<String> {
            self.0.decode(ids, skip)
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.0.token_to_id(token)
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            self.0.id_to_token(id)
        }
    }

    struct Backend(DynTokenizer);

    impl TextBackend for Backend {
        fn tokenizer(&self) -> DynTokenizer {
            self.0.clone()
        }

        fn model_id(&self) -> &str {
            "bert-like"
        }

        fn pad_token_id(&self) -> Option<u32> {
            Some(0)
        }
    }

    fn processor() -> TextRequestProcessor {
        let tokenizer = BertLikeTokenizer(TestTokenizer::new().with_vocab_size(512));
        TextRequestProcessor::new(Arc::new(Backend(Arc::new(tokenizer))), 8)
    }

    fn text(value: &str) -> Prompt {
        Prompt::Text(value.to_string())
    }

    #[test]
    fn default_side_truncation_keeps_special_tokens_like_hf() {
        let params = PoolingTokenization {
            truncate_prompt_tokens: Some(4),
            ..PoolingTokenization::new(Some(8), true)
        };
        // [CLS] a b [SEP]: the tokenizer drops content, not the trailing SEP.
        assert_eq!(
            processor().tokenize_pooling_prompt(text("abcdef"), &params).unwrap(),
            vec![301, 97, 98, 302]
        );
    }

    #[test]
    fn explicit_side_truncation_slices_the_encoded_prompt() {
        let right = PoolingTokenization {
            truncate_prompt_tokens: Some(3),
            truncation_side: Some(TruncationSide::Right),
            ..PoolingTokenization::new(Some(8), true)
        };
        let left = PoolingTokenization {
            truncation_side: Some(TruncationSide::Left),
            ..right.clone()
        };
        let processor = processor();
        assert_eq!(
            processor.tokenize_pooling_prompt(text("abcd"), &right).unwrap(),
            vec![301, 97, 98]
        );
        assert_eq!(
            processor.tokenize_pooling_prompt(text("abcd"), &left).unwrap(),
            vec![99, 100, 302]
        );
    }

    #[test]
    fn full_budget_truncation_keeps_every_input_token() {
        // Unlike generation, pooling does not hold a token back for output.
        let params = PoolingTokenization {
            truncate_prompt_tokens: Some(-1),
            truncation_side: Some(TruncationSide::Left),
            ..PoolingTokenization::new(Some(8), true)
        };
        let tokens = processor()
            .tokenize_pooling_prompt(Prompt::TokenIds((0..20).collect()), &params)
            .unwrap();
        assert_eq!(tokens, (12..20).collect::<Vec<_>>());
    }

    #[test]
    fn padding_lowercase_and_length_errors_match_python() {
        let processor = processor();
        let padded = PoolingTokenization {
            pad_prompt_tokens: Some(6),
            do_lower_case: true,
            ..PoolingTokenization::new(Some(8), true)
        };
        assert_eq!(
            processor.tokenize_pooling_prompt(text("AB"), &padded).unwrap(),
            vec![301, 97, 98, 302, 0, 0]
        );

        let limited = PoolingTokenization::new(Some(4), true);
        let text_error = processor.tokenize_pooling_prompt(text("abcdefgh"), &limited).unwrap_err();
        let ids_error = processor
            .tokenize_pooling_prompt(Prompt::TokenIds(vec![1; 7]), &limited)
            .unwrap_err();
        let truncation_error = processor
            .tokenize_pooling_prompt(
                text("a"),
                &PoolingTokenization {
                    truncate_prompt_tokens: Some(5),
                    ..limited.clone()
                },
            )
            .unwrap_err();
        assert_eq!(
            [
                text_error.to_report_string(),
                ids_error.to_report_string(),
                truncation_error.to_report_string(),
            ],
            [
                "This model's maximum context length is 4 tokens. However, your prompt \
                 contains at least 5 input tokens, which exceeds this limit even without any \
                 output tokens. Please reduce the length of the input prompt.",
                "This model's maximum context length is 4 tokens. However, your prompt \
                 contains 7 input tokens, which exceeds this limit even without any output \
                 tokens. Please reduce the length of the input prompt.",
                "truncate_prompt_tokens=5 cannot be greater than max_model_len = 4. Please \
                 request a smaller truncation size.",
            ]
        );
    }

    #[test]
    fn unlimited_prompts_skip_length_checks() {
        let params = PoolingTokenization::new(None, false);
        let tokens = processor()
            .tokenize_pooling_prompt(Prompt::TokenIds((0..100).collect()), &params)
            .unwrap();
        assert_eq!(tokens.len(), 100);
    }
}
