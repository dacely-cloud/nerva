use crate::common::math::silu;
use crate::common::shape::TransformerBlockShape;
use crate::reference::block::ReferenceTransformerBlock;
use crate::reference::scratch::TransformerBlockScratch;
use crate::reference::smoke::{ReferenceBlockSmokeStatus, reference_block_smoke};
use nerva_ledger::types::token::ledger::TokenLedger;

#[test]
fn zero_block_preserves_residual() {
    let shape = TransformerBlockShape::new(4, 2, 8);
    let block = ReferenceTransformerBlock::zero_for_shape(shape).unwrap();
    let mut scratch = TransformerBlockScratch::new(shape).unwrap();
    let mut output = [0.0; 4];
    let input = [1.0, -2.0, 3.0, -4.0];
    let mut ledger = TokenLedger::new(0);

    block
        .forward_into(&input, &mut scratch, &mut output, &mut ledger)
        .unwrap();

    assert_eq!(output, input);
    assert_eq!(ledger.hot_path_allocations, 0);
    assert!(ledger.require_zero_hot_path_allocations().is_ok());
}

#[test]
fn nontrivial_block_matches_hand_reference() {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let block = ReferenceTransformerBlock::new(
        shape,
        vec![1.0, 1.0],
        vec![1.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.5, 0.0, 0.0, 0.5],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        1e-5,
    )
    .unwrap();
    let mut scratch = TransformerBlockScratch::new(shape).unwrap();
    let mut output = [0.0; 2];
    let input = [1.0, 2.0];
    let mut ledger = TokenLedger::new(7);

    block
        .forward_into(&input, &mut scratch, &mut output, &mut ledger)
        .unwrap();

    let attn_norm_scale = ((1.0_f32 + 4.0) / 2.0 + 1e-5).sqrt().recip();
    let attn = [input[0] * attn_norm_scale, input[1] * attn_norm_scale];
    let residual = [input[0] + attn[0], input[1] + attn[1]];
    let mlp_norm_scale = ((residual[0] * residual[0] + residual[1] * residual[1]) / 2.0 + 1e-5)
        .sqrt()
        .recip();
    let mlp_norm = [residual[0] * mlp_norm_scale, residual[1] * mlp_norm_scale];
    let expected = [
        residual[0] + silu(0.5 * mlp_norm[0]) * mlp_norm[0],
        residual[1] + silu(0.5 * mlp_norm[1]) * mlp_norm[1],
    ];

    for (actual, expected) in output.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(ledger.hot_path_allocations, 0);
}

#[test]
fn rejects_bad_shapes_and_scratch_mismatch() {
    assert!(TransformerBlockShape::new(3, 2, 4).validate().is_err());
    let block =
        ReferenceTransformerBlock::zero_for_shape(TransformerBlockShape::new(4, 2, 8)).unwrap();
    let mut scratch = TransformerBlockScratch::new(TransformerBlockShape::new(2, 1, 2)).unwrap();
    let mut ledger = TokenLedger::new(0);
    let mut output = [0.0; 4];
    assert!(
        block
            .forward_into(&[0.0; 4], &mut scratch, &mut output, &mut ledger)
            .is_err()
    );
}

#[test]
fn reference_block_smoke_reports_hash_and_no_allocations() {
    let summary = reference_block_smoke().unwrap();
    assert_eq!(summary.status, ReferenceBlockSmokeStatus::Ok);
    assert_eq!(summary.hidden, 2);
    assert_eq!(summary.heads, 1);
    assert_eq!(summary.intermediate, 2);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(summary.output_hash, 3_850_145_622_605_741_247);
    assert!(summary.to_json().contains("\"status\":\"ok\""));
}
