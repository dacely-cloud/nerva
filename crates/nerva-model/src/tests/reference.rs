use crate::common::math::{sigmoid, silu};
use crate::common::shape::TransformerBlockShape;
use crate::reference::block::types::ReferenceTransformerBlock;
use crate::reference::router::{
    DeepSeekRouterScoring, DeepSeekV3GroupedRouterConfig, DeepSeekV4RouterConfig,
    deepseek_v3_grouped_route, deepseek_v4_sqrtsoftplus_route,
};
use crate::reference::scratch::types::TransformerBlockScratch;
use crate::reference::smoke::run::reference_block_smoke;
use crate::reference::smoke::status::ReferenceBlockSmokeStatus;
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
fn deepseek_v3_grouped_router_uses_bias_for_choice_not_weight() {
    let logits = [-2.0, 0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -3.0];
    let correction_bias = [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, -4.0, 0.0];
    let route = deepseek_v3_grouped_route(
        &logits,
        Some(&correction_bias),
        DeepSeekV3GroupedRouterConfig {
            top_k: 2,
            num_expert_groups: 2,
            top_k_groups: 1,
            scoring: DeepSeekRouterScoring::Sigmoid,
            renormalize: true,
            routed_scaling_factor: 2.5,
        },
    )
    .unwrap();

    assert_eq!(route.expert_ids, vec![3, 2]);
    let raw3 = sigmoid(logits[3]);
    let raw2 = sigmoid(logits[2]);
    let sum = raw3 + raw2;
    assert!((route.weights[0] - raw3 / sum * 2.5).abs() < 1e-6);
    assert!((route.weights[1] - raw2 / sum * 2.5).abs() < 1e-6);
}

#[test]
fn deepseek_v4_router_uses_sqrtsoftplus_bias_for_choice() {
    let logits = [-2.0, 0.0, 1.0, 3.0];
    let correction_bias = [0.0, 3.0, 0.0, -3.0];
    let route = deepseek_v4_sqrtsoftplus_route(
        &logits,
        Some(&correction_bias),
        None,
        DeepSeekV4RouterConfig {
            top_k: 2,
            renormalize: true,
            routed_scaling_factor: 1.5,
        },
    )
    .unwrap();

    assert_eq!(route.expert_ids, vec![1, 2]);
    let raw1 = softplus(logits[1]).sqrt();
    let raw2 = softplus(logits[2]).sqrt();
    let sum = raw1 + raw2;
    assert!((route.weights[0] - raw1 / sum * 1.5).abs() < 1e-6);
    assert!((route.weights[1] - raw2 / sum * 1.5).abs() < 1e-6);
}

#[test]
fn deepseek_v4_hash_router_uses_table_ids_and_unbiased_weights() {
    let logits = [4.0, -1.0, 0.0, 2.0];
    let hash_ids = [2usize, 1usize, 3usize];
    let route = deepseek_v4_sqrtsoftplus_route(
        &logits,
        Some(&[99.0, 99.0, 99.0, 99.0]),
        Some(&hash_ids),
        DeepSeekV4RouterConfig {
            top_k: 3,
            renormalize: true,
            routed_scaling_factor: 1.0,
        },
    )
    .unwrap();

    assert_eq!(route.expert_ids, hash_ids);
    let raw = hash_ids
        .iter()
        .map(|id| softplus(logits[*id]).sqrt())
        .collect::<Vec<_>>();
    let sum = raw.iter().sum::<f32>();
    for (actual, expected) in route.weights.iter().zip(raw.iter()) {
        assert!((actual - expected / sum).abs() < 1e-6);
    }
}

#[test]
fn deepseek_router_rejects_invalid_configs() {
    assert!(
        deepseek_v3_grouped_route(
            &[0.0, 1.0, 2.0],
            None,
            DeepSeekV3GroupedRouterConfig {
                top_k: 1,
                num_expert_groups: 2,
                top_k_groups: 1,
                scoring: DeepSeekRouterScoring::Sigmoid,
                renormalize: true,
                routed_scaling_factor: 1.0,
            },
        )
        .is_err()
    );

    assert!(
        deepseek_v4_sqrtsoftplus_route(
            &[0.0, 1.0],
            None,
            Some(&[2]),
            DeepSeekV4RouterConfig {
                top_k: 1,
                renormalize: true,
                routed_scaling_factor: 1.0,
            },
        )
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

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}
