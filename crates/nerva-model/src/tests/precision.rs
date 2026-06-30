use crate::common::rope::apply_rotary_to_query_key;
use crate::common::shape::TransformerBlockShape;
use crate::precision::bits::{
    bf16_bits_to_f32, dequantize_f8_e4m3fn_block_scaled, dequantize_f8_e4m3fn_block_scaled_into,
    dequantize_mxfp4_e2m1_block_scaled, dequantize_mxfp4_e2m1_block_scaled_into,
    e8m0_exponent_bits_to_f32, f8_e4m3fn_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits,
    f32_to_f16_bits, mxfp4_e2m1_nibble_to_f32,
};
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::precision::file_smoke::run::precision_block_from_safetensors_smoke;
use crate::precision::file_smoke::summary::PrecisionSafetensorsBlockSmokeStatus;
use crate::precision::scratch::PrecisionTransformerBlockScratch;
use crate::precision::smoke::run::precision_block_smoke;
use crate::precision::smoke::status::PrecisionBlockSmokeStatus;
use nerva_core::types::dtype::DType;
use nerva_ledger::types::token::ledger::TokenLedger;

#[test]
fn f16_and_bf16_conversions_round_known_values() {
    assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
    assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
    assert_eq!(f32_to_f16_bits(0.5), 0x3800);
    assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
    assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
    assert_eq!(f16_bits_to_f32(0xc000), -2.0);

    assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
    assert_eq!(f32_to_bf16_bits(-2.0), 0xc000);
    assert_eq!(f32_to_bf16_bits(0.5), 0x3f00);
    assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
    assert_eq!(bf16_bits_to_f32(0xc000), -2.0);
}

#[test]
fn fp8_e4m3fn_conversion_matches_torch_reference_bytes() {
    let cases = [
        (0x00, 0.0),
        (0x01, 0.001953125),
        (0x07, 0.013671875),
        (0x08, 0.015625),
        (0x20, 0.125),
        (0x30, 0.5),
        (0x38, 1.0),
        (0x40, 2.0),
        (0x70, 128.0),
        (0x77, 240.0),
        (0x78, 256.0),
        (0x7e, 448.0),
        (0xb8, -1.0),
        (0xf8, -256.0),
    ];

    for (bits, expected) in cases {
        assert_eq!(f8_e4m3fn_bits_to_f32(bits), expected, "bits {bits:#04x}");
    }

    assert!(f8_e4m3fn_bits_to_f32(0x7f).is_nan());
    assert!(f8_e4m3fn_bits_to_f32(0xff).is_nan());
    assert_eq!(f8_e4m3fn_bits_to_f32(0x80).to_bits(), (-0.0f32).to_bits());
}

#[test]
fn e8m0_scale_upcast_matches_vllm_raw_exponent_path() {
    let cases = [
        (0x00, 0.0),
        (0x01, f32::from_bits(0x0080_0000)),
        (0x7e, 0.5),
        (0x7f, 1.0),
        (0x80, 2.0),
        (0xb8, f32::from_bits(0x5c00_0000)),
    ];

    for (bits, expected) in cases {
        assert_eq!(
            e8m0_exponent_bits_to_f32(bits),
            expected,
            "bits {bits:#04x}"
        );
    }

    assert!(e8m0_exponent_bits_to_f32(0xff).is_infinite());
}

#[test]
fn fp8_block_dequantization_applies_e8m0_scales_per_tile() {
    let weights = [
        0x38, 0x40, 0x30, 0xb8, //
        0x70, 0x77, 0x78, 0x7e, //
        0x20, 0x28, 0x30, 0x38,
    ];
    let scales = [
        0x7f, 0x80, //
        0x7e, 0x81,
    ];

    let values = dequantize_f8_e4m3fn_block_scaled(&weights, &scales, 3, 4, 2, 2).unwrap();

    assert_eq!(
        values,
        vec![
            1.0, 2.0, 1.0, -2.0, //
            128.0, 240.0, 512.0, 896.0, //
            0.0625, 0.125, 2.0, 4.0,
        ]
    );
}

#[test]
fn fp8_block_dequantization_rejects_bad_shapes() {
    let weights = [0x38, 0x38, 0x38, 0x38];
    let scales = [0x7f];
    let mut output = [0.0; 4];

    assert!(
        dequantize_f8_e4m3fn_block_scaled_into(&weights, &scales, 2, 2, 0, 2, &mut output).is_err()
    );
    assert!(
        dequantize_f8_e4m3fn_block_scaled_into(&weights[..3], &scales, 2, 2, 2, 2, &mut output)
            .is_err()
    );
    assert!(
        dequantize_f8_e4m3fn_block_scaled_into(&weights, &[], 2, 2, 1, 1, &mut output).is_err()
    );
}

#[test]
fn mxfp4_e2m1_nibble_conversion_matches_vllm_lut() {
    let expected = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];

    for (nibble, expected) in expected.into_iter().enumerate() {
        assert_eq!(
            mxfp4_e2m1_nibble_to_f32(nibble as u8),
            expected,
            "nibble {nibble:#x}"
        );
    }
}

#[test]
fn mxfp4_block_dequantization_unpacks_low_nibble_first() {
    let packed = [
        0x21, 0x76, 0xa9, 0xfe, //
        0x10, 0x54, 0x98, 0xdc,
    ];
    let scales = [
        0x7f, 0x80, //
        0x7e, 0x81,
    ];

    let values = dequantize_mxfp4_e2m1_block_scaled(&packed, &scales, 2, 4, 2).unwrap();

    assert_eq!(
        values,
        vec![
            0.5, 1.0, 4.0, 6.0, -1.0, -2.0, -8.0, -12.0, //
            0.0, 0.25, 1.0, 1.5, -0.0, -2.0, -8.0, -12.0,
        ]
    );
}

#[test]
fn mxfp4_block_dequantization_rejects_bad_shapes() {
    let packed = [0x21, 0x43];
    let scales = [0x7f];
    let mut output = [0.0; 4];

    assert!(
        dequantize_mxfp4_e2m1_block_scaled_into(&packed, &scales, 1, 2, 0, &mut output).is_err()
    );
    assert!(
        dequantize_mxfp4_e2m1_block_scaled_into(&packed[..1], &scales, 1, 2, 2, &mut output)
            .is_err()
    );
    assert!(dequantize_mxfp4_e2m1_block_scaled_into(&packed, &[], 1, 2, 1, &mut output).is_err());
}

#[test]
fn precision_block_smoke_reports_f16_and_bf16_bit_parity() {
    let summary = precision_block_smoke().unwrap();

    assert_eq!(summary.status, PrecisionBlockSmokeStatus::Ok);
    assert!(summary.passed());
    assert!(summary.f16.bit_parity);
    assert!(summary.bf16.bit_parity);
    assert_eq!(summary.f16.hot_path_allocations, 0);
    assert_eq!(summary.bf16.hot_path_allocations, 0);
    assert_eq!(summary.f16.output_hash, summary.f16.expected_hash);
    assert_eq!(summary.bf16.output_hash, summary.bf16.expected_hash);
    assert!(summary.to_json().contains("\"dtype\":\"float16\""));
    assert!(summary.to_json().contains("\"dtype\":\"bfloat16\""));
}

#[test]
fn precision_block_loads_weights_from_safetensors_payload() {
    let summary = precision_block_from_safetensors_smoke().unwrap();

    assert_eq!(summary.status, PrecisionSafetensorsBlockSmokeStatus::Ok);
    assert!(summary.passed());
    assert_eq!(summary.tensors_loaded, 9);
    assert_eq!(summary.bytes_loaded, 64);
    assert_ne!(summary.data_hash, 0);
    assert_eq!(summary.output_hash, summary.expected_hash);
    assert!(summary.bit_parity);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"tensors_loaded\":9"));
}

#[test]
fn precision_block_rejects_non_16_bit_dtypes() {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let rms = [1.0, 1.0];
    let identity = [1.0, 0.0, 0.0, 1.0];

    assert!(
        PrecisionTransformerBlock::new_from_f32(
            DType::F32,
            shape,
            &rms,
            &rms,
            &identity,
            &identity,
            &identity,
            &identity,
            &identity,
            &identity,
            &identity,
            1e-5,
        )
        .is_err()
    );
}

#[test]
fn precision_block_rejects_scratch_shape_mismatch() {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let rms = [1.0, 1.0];
    let identity = [1.0, 0.0, 0.0, 1.0];
    let block = PrecisionTransformerBlock::new_from_f32(
        DType::F16,
        shape,
        &rms,
        &rms,
        &identity,
        &identity,
        &identity,
        &identity,
        &identity,
        &identity,
        &identity,
        1e-5,
    )
    .unwrap();
    let mut scratch =
        PrecisionTransformerBlockScratch::new(TransformerBlockShape::new(4, 2, 4)).unwrap();
    let input = [f32_to_f16_bits(1.0), f32_to_f16_bits(2.0)];
    let mut output = [0u16; 2];
    let mut ledger = TokenLedger::new(0);

    assert!(
        block
            .forward_into(&input, &mut scratch, &mut output, &mut ledger)
            .is_err()
    );
}

#[test]
fn precision_block_accepts_grouped_query_kv_projection_shapes() {
    let shape = TransformerBlockShape::new_with_kv_heads(4, 2, 1, 4);
    let rms = [1.0, 1.0, 1.0, 1.0];
    let full_identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let compact_kv = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let block = PrecisionTransformerBlock::new_from_f32(
        DType::F16,
        shape,
        &rms,
        &rms,
        &full_identity,
        &compact_kv,
        &compact_kv,
        &full_identity,
        &full_identity,
        &full_identity,
        &full_identity,
        1e-5,
    )
    .unwrap();
    let mut scratch = PrecisionTransformerBlockScratch::new(shape).unwrap();
    let input = [
        f32_to_f16_bits(1.0),
        f32_to_f16_bits(0.0),
        f32_to_f16_bits(0.0),
        f32_to_f16_bits(1.0),
    ];
    let mut output = [0u16; 4];
    let mut ledger = TokenLedger::new(0);

    block
        .forward_into(&input, &mut scratch, &mut output, &mut ledger)
        .unwrap();

    assert!(output.iter().any(|value| *value != 0));
    assert_eq!(ledger.hot_path_allocations, 0);
}

#[test]
fn rotary_embedding_rotates_query_and_compact_key_heads() {
    let shape = TransformerBlockShape::new_with_kv_heads(4, 1, 1, 4);
    let mut query = [1.0, 2.0, 3.0, 4.0];
    let mut key = [0.5, -1.0, -0.25, 0.75];
    let angle0 = 1.0f32;
    let angle1 = 0.01f32;
    let (sin0, cos0) = angle0.sin_cos();
    let (sin1, cos1) = angle1.sin_cos();

    apply_rotary_to_query_key(shape, 1, 10_000.0, &mut query, &mut key).unwrap();

    let expected_query = [
        1.0 * cos0 - 3.0 * sin0,
        2.0 * cos1 - 4.0 * sin1,
        3.0 * cos0 + 1.0 * sin0,
        4.0 * cos1 + 2.0 * sin1,
    ];
    let expected_key = [
        0.5 * cos0 - -0.25 * sin0,
        -cos1 - 0.75 * sin1,
        -0.25 * cos0 + 0.5 * sin0,
        0.75 * cos1 + -1.0 * sin1,
    ];
    for (actual, expected) in query.iter().zip(expected_query.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    for (actual, expected) in key.iter().zip(expected_key.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}
