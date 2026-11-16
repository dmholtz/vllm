// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Output encodings shared by the pooling endpoints.
//!
//! Mirrors Python `vllm/entrypoints/pooling/utils.py` and
//! `vllm/utils/serial_utils.py` (`tensor2binary`, `EMBED_DTYPES`).

use axum::body::Body;
use axum::http::{HeaderValue, header};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

use crate::error::{ApiError, server_error};

/// Response encoding of pooling outputs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EncodingFormat {
    /// JSON numbers.
    #[default]
    Float,
    /// Base64 of the raw `embed_dtype` bytes, in JSON.
    Base64,
    /// Raw bytes with a JSON `metadata` response header.
    Bytes,
    /// Raw bytes without metadata.
    BytesOnly,
}

impl EncodingFormat {
    /// Whether the response body is raw bytes instead of JSON.
    pub(crate) fn is_binary(self) -> bool {
        matches!(self, Self::Bytes | Self::BytesOnly)
    }
}

/// Numeric type of base64 and binary pooling outputs (Python `EmbedDType`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EmbedDType {
    #[default]
    Float32,
    Float16,
    Bfloat16,
    #[serde(rename = "fp8_e4m3")]
    Fp8E4m3,
    #[serde(rename = "fp8_e5m2")]
    Fp8E5m2,
}

impl EmbedDType {
    fn element_size(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 | Self::Bfloat16 => 2,
            Self::Fp8E4m3 | Self::Fp8E5m2 => 1,
        }
    }
}

/// Byte order of base64 and binary pooling outputs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Endianness {
    #[default]
    Native,
    Big,
    Little,
}

impl Endianness {
    fn needs_swap(self) -> bool {
        match self {
            Self::Native => false,
            Self::Little => cfg!(target_endian = "big"),
            Self::Big => cfg!(target_endian = "little"),
        }
    }
}

/// Round `value` to the nearest representable float8 with `MANTISSA_BITS`
/// mantissa bits and exponent `BIAS`, ties to even. Returns the unsigned
/// magnitude bits, or `None` if the value overflows the finite range.
fn float8_magnitude<const MANTISSA_BITS: i32, const BIAS: i32>(
    magnitude: f64,
    max_exponent_field: i32,
) -> Option<u8> {
    if magnitude == 0.0 {
        return Some(0);
    }
    if !magnitude.is_finite() {
        return None;
    }
    let min_normal_exponent = 1 - BIAS;
    let (mut exponent_field, mantissa) = if magnitude < 2f64.powi(min_normal_exponent) {
        // Subnormal: multiples of 2^(min_normal_exponent - MANTISSA_BITS).
        let scaled = magnitude * 2f64.powi(MANTISSA_BITS - min_normal_exponent);
        let mantissa = scaled.round_ties_even() as i32;
        // Rounding up to 1 << MANTISSA_BITS yields the smallest normal value.
        (
            mantissa >> MANTISSA_BITS,
            mantissa & ((1 << MANTISSA_BITS) - 1),
        )
    } else {
        let exponent = magnitude.log2().floor() as i32;
        // Guard against log2 rounding at exact powers of two.
        let exponent = if 2f64.powi(exponent) > magnitude {
            exponent - 1
        } else if 2f64.powi(exponent + 1) <= magnitude {
            exponent + 1
        } else {
            exponent
        };
        let scaled = (magnitude / 2f64.powi(exponent) - 1.0) * 2f64.powi(MANTISSA_BITS);
        let mut mantissa = scaled.round_ties_even() as i32;
        let mut exponent = exponent;
        if mantissa == 1 << MANTISSA_BITS {
            mantissa = 0;
            exponent += 1;
        }
        (exponent + BIAS, mantissa)
    };
    if exponent_field > max_exponent_field {
        return None;
    }
    exponent_field = exponent_field.max(0);
    Some(((exponent_field << MANTISSA_BITS) | mantissa) as u8)
}

/// Torch `float32 -> float8_e4m3fn` cast: round to nearest even, saturating
/// to ±448 (including infinities); NaN maps to `0x7f`.
fn f32_to_fp8_e4m3fn(value: f32) -> u8 {
    const MAX_FINITE: u8 = 0x7e;
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let bits = float8_magnitude::<3, 7>(f64::from(value.abs()), 15)
        // Exponent field 15 with mantissa 0b111 is NaN in the `fn` variant.
        .filter(|bits| *bits <= MAX_FINITE)
        .unwrap_or(MAX_FINITE);
    sign | bits
}

/// Torch `float32 -> float8_e5m2` cast: round to nearest even, overflowing
/// to infinity; NaN maps to `0x7f`.
fn f32_to_fp8_e5m2(value: f32) -> u8 {
    const INFINITY: u8 = 0x7c;
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    if value.is_infinite() {
        return sign | INFINITY;
    }
    sign | float8_magnitude::<2, 15>(f64::from(value.abs()), 30).unwrap_or(INFINITY)
}

/// Python `tensor2binary`: cast to `embed_dtype`, then byte-swap each element
/// when the requested byte order differs from the host's.
pub(crate) fn encode_values(values: &[f32], dtype: EmbedDType, endianness: Endianness) -> Vec<u8> {
    let mut bytes: Vec<u8> = match dtype {
        EmbedDType::Float32 => values.iter().flat_map(|value| value.to_ne_bytes()).collect(),
        EmbedDType::Float16 => {
            values.iter().flat_map(|value| f16::from_f32(*value).to_ne_bytes()).collect()
        }
        EmbedDType::Bfloat16 => {
            values.iter().flat_map(|value| bf16::from_f32(*value).to_ne_bytes()).collect()
        }
        EmbedDType::Fp8E4m3 => values.iter().map(|value| f32_to_fp8_e4m3fn(*value)).collect(),
        EmbedDType::Fp8E5m2 => values.iter().map(|value| f32_to_fp8_e5m2(*value)).collect(),
    };
    let element_size = dtype.element_size();
    if element_size > 1 && endianness.needs_swap() {
        bytes.chunks_exact_mut(element_size).for_each(<[u8]>::reverse);
    }
    bytes
}

/// Base64-encode values like Python `encode_pooling_output_base64`.
pub(crate) fn encode_base64(values: &[f32], dtype: EmbedDType, endianness: Endianness) -> String {
    STANDARD.encode(encode_values(values, dtype, endianness))
}

/// Usage of a pooling response, mirroring Python `UsageInfo` as serialized by
/// the pooling entrypoints: no completion tokens and no token details.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct PoolingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
    pub completion_tokens: usize,
    pub prompt_tokens_details: Option<()>,
    pub completion_tokens_details: Option<()>,
}

impl PoolingUsage {
    pub(crate) fn new(prompt_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            total_tokens: prompt_tokens,
            completion_tokens: 0,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }
    }
}

/// One output of a binary pooling response.
pub(crate) struct BinaryItem<'a> {
    pub values: &'a [f32],
    pub shape: &'a [usize],
}

#[derive(Serialize)]
struct BinaryItemMetadata<'a> {
    index: usize,
    embed_dtype: EmbedDType,
    endianness: Endianness,
    start: usize,
    end: usize,
    shape: &'a [usize],
}

#[derive(Serialize)]
struct BinaryUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
struct BinaryMetadata<'a> {
    id: &'a str,
    created: u64,
    model: &'a str,
    data: Vec<BinaryItemMetadata<'a>>,
    usage: BinaryUsage,
}

/// Response metadata of a binary pooling response.
pub(crate) struct BinaryResponseInfo<'a> {
    pub id: &'a str,
    pub created: u64,
    pub model: &'a str,
    pub prompt_tokens: usize,
}

/// Python `build_pooling_bytes_streaming_response`: concatenated raw outputs
/// as `application/octet-stream`, with a JSON `metadata` header describing
/// each output's byte range unless `bytes_only` was requested.
pub(crate) fn binary_response(
    items: &[BinaryItem<'_>],
    info: BinaryResponseInfo<'_>,
    encoding_format: EncodingFormat,
    dtype: EmbedDType,
    endianness: Endianness,
) -> Result<Response, ApiError> {
    let mut body = Vec::new();
    let mut metadata = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let start = body.len();
        body.extend(encode_values(item.values, dtype, endianness));
        metadata.push(BinaryItemMetadata {
            index,
            embed_dtype: dtype,
            endianness,
            start,
            end: body.len(),
            shape: item.shape,
        });
    }

    let mut response = Body::from(body).into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    if encoding_format == EncodingFormat::Bytes {
        let metadata = serde_json::to_string(&BinaryMetadata {
            id: info.id,
            created: info.created,
            model: info.model,
            data: metadata,
            usage: BinaryUsage {
                prompt_tokens: info.prompt_tokens,
                total_tokens: info.prompt_tokens,
            },
        })
        .map_err(|error| server_error!("failed to encode response metadata: {error}"))?;
        let value = HeaderValue::from_str(&metadata)
            .map_err(|error| server_error!("invalid response metadata header: {error}"))?;
        headers.insert("metadata", value);
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// Inputs and little-endian outputs of Python `tensor2binary` (torch
    /// casts) for every reduced-precision `embed_dtype`.
    #[test]
    fn dtype_encodings_match_python_tensor2binary() {
        let values = [
            0.0,
            -0.0,
            1.0,
            -2.5,
            0.1,
            0.333_333_3,
            1e-3,
            1e-7,
            3e-3,
            447.9,
            448.0,
            460.0,
            470.0,
            500.0,
            57344.0,
            60000.0,
            70000.0,
            -1e-9,
            0.000_976_562_5,
            0.015_625 * 1.0625,
            f32::INFINITY,
            f32::NAN,
        ];
        let cases = [
            (
                EmbedDType::Float16,
                "00000080003c00c1662e553519140200251a005f005f305f585fd05f007b537b007c008000144024007c007e",
            ),
            (
                EmbedDType::Bfloat16,
                "00000080803f20c0cd3dab3e833ad733453be043e043e643eb43fa4360476a47894789b0803a883c807fc07f",
            ),
            (
                EmbedDType::Fp8E4m3,
                "008038c21d2b0100027e7e7e7e7e7e7e7e8000087e7f",
            ),
            (
                EmbedDType::Fp8E5m2,
                "00803cc12e3514001a5f5f5f5f607b7b7c8014247c7f",
            ),
        ];
        for (dtype, expected) in cases {
            assert_eq!(
                hex(&encode_values(&values, dtype, Endianness::Little)),
                expected,
                "{dtype:?}"
            );
        }
    }

    #[test]
    fn byte_order_swaps_multi_byte_elements_only() {
        assert_eq!(
            hex(&encode_values(
                &[1.0, -2.5],
                EmbedDType::Bfloat16,
                Endianness::Big
            )),
            "3f80c020"
        );
        assert_eq!(
            hex(&encode_values(
                &[1.0, -2.5],
                EmbedDType::Float32,
                Endianness::Big
            )),
            "3f800000c0200000"
        );
        assert_eq!(
            encode_values(&[1.0], EmbedDType::Fp8E4m3, Endianness::Big),
            encode_values(&[1.0], EmbedDType::Fp8E4m3, Endianness::Little)
        );
    }
}
