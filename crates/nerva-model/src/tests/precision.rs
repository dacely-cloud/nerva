use crate::common::shape::TransformerBlockShape;
use crate::precision::bits::{
    bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits,
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
