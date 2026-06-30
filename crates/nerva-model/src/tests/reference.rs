use crate::common::math::{sigmoid, silu};
use crate::common::shape::TransformerBlockShape;
use crate::precision::block::moe::{
    PrecisionMoeConfig, PrecisionMoeRouterKind, select_moe_route_for_logits,
    select_moe_route_for_logits_with_hash_token,
};
use crate::reference::block::types::ReferenceTransformerBlock;
use crate::reference::moe::{
    DeepSeekRoutedMoeConfig, deepseek_routed_moe_forward, deepseek_swiglu,
};
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
fn precision_moe_deepseek_v3_router_matches_reference() {
    let logits = [-2.0, 0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -3.0];
    let correction_bias = [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, -4.0, 0.0];
    let precision_route = select_moe_route_for_logits(
        &logits,
        &correction_bias,
        PrecisionMoeConfig {
            moe_intermediate: 2,
            shared_expert_intermediate: 0,
            num_experts: 8,
            experts_per_token: 2,
            norm_topk_prob: true,
            router_kind: PrecisionMoeRouterKind::DeepSeekV3GroupedSigmoid {
                num_expert_groups: 2,
                top_k_groups: 1,
                routed_scaling_factor: 2.5,
            },
        },
    )
    .unwrap();
    let reference_route = deepseek_v3_grouped_route(
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

    assert_eq!(
        precision_route
            .iter()
            .map(|(expert, _)| *expert)
            .collect::<Vec<_>>(),
        reference_route.expert_ids
    );
    for ((_, actual), expected) in precision_route.iter().zip(reference_route.weights.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn precision_moe_deepseek_v4_router_matches_reference() {
    let logits = [-2.0, 0.0, 1.0, 3.0];
    let correction_bias = [0.0, 3.0, 0.0, -3.0];
    let precision_route = select_moe_route_for_logits(
        &logits,
        &correction_bias,
        PrecisionMoeConfig {
            moe_intermediate: 2,
            shared_expert_intermediate: 0,
            num_experts: 4,
            experts_per_token: 2,
            norm_topk_prob: true,
            router_kind: PrecisionMoeRouterKind::DeepSeekV4SqrtSoftplus {
                routed_scaling_factor: 1.5,
            },
        },
    )
    .unwrap();
    let reference_route = deepseek_v4_sqrtsoftplus_route(
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

    assert_eq!(
        precision_route
            .iter()
            .map(|(expert, _)| *expert)
            .collect::<Vec<_>>(),
        reference_route.expert_ids
    );
    for ((_, actual), expected) in precision_route.iter().zip(reference_route.weights.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn precision_moe_deepseek_v4_hash_router_matches_reference() {
    let logits = [4.0, -1.0, 0.0, 2.0];
    let hash_table = [
        0usize, 1, 3, // token 0
        2, 1, 3, // token 1
        3, 0, 2, // token 2
    ];
    let precision_route = select_moe_route_for_logits_with_hash_token(
        &logits,
        &[99.0, 99.0, 99.0, 99.0],
        &hash_table,
        Some(1),
        PrecisionMoeConfig {
            moe_intermediate: 2,
            shared_expert_intermediate: 0,
            num_experts: 4,
            experts_per_token: 3,
            norm_topk_prob: true,
            router_kind: PrecisionMoeRouterKind::DeepSeekV4Hash {
                routed_scaling_factor: 1.0,
            },
        },
    )
    .unwrap();
    let reference_route = deepseek_v4_sqrtsoftplus_route(
        &logits,
        Some(&[99.0, 99.0, 99.0, 99.0]),
        Some(&[2usize, 1usize, 3usize]),
        DeepSeekV4RouterConfig {
            top_k: 3,
            renormalize: true,
            routed_scaling_factor: 1.0,
        },
    )
    .unwrap();

    assert_eq!(
        precision_route
            .iter()
            .map(|(expert, _)| *expert)
            .collect::<Vec<_>>(),
        reference_route.expert_ids
    );
    for ((_, actual), expected) in precision_route.iter().zip(reference_route.weights.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn precision_moe_deepseek_v4_hash_router_requires_route_table() {
    let error = select_moe_route_for_logits(
        &[4.0, -1.0, 0.0, 2.0],
        &[0.0, 0.0, 0.0, 0.0],
        PrecisionMoeConfig {
            moe_intermediate: 2,
            shared_expert_intermediate: 0,
            num_experts: 4,
            experts_per_token: 2,
            norm_topk_prob: true,
            router_kind: PrecisionMoeRouterKind::DeepSeekV4Hash {
                routed_scaling_factor: 1.0,
            },
        },
    )
    .unwrap_err();
    assert!(format!("{error:?}").contains("tid2eid route table"));
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
fn deepseek_routed_moe_combines_selected_experts_with_swiglu_clamp() {
    let input = [1.2, -0.7, 0.3];
    let expert_ids = [1usize, 0usize];
    let expert_weights = [0.75, 0.25];
    let w_gate = [
        1.0, -0.5, 0.25, -0.25, 0.75, 1.25, 0.5, 0.2, -0.1, -1.0, 0.4, 0.3,
    ];
    let w_up = [
        -0.2, 0.4, 1.1, 0.8, -0.6, 0.2, 1.5, -0.3, 0.1, 0.7, 0.6, -0.4,
    ];
    let w_down = [
        0.3, -0.2, 0.4, 0.1, -0.5, 0.2, -0.7, 0.6, -0.1, 0.25, 0.35, -0.45,
    ];
    let mut output = [0.0; 3];

    deepseek_routed_moe_forward(
        &input,
        &expert_ids,
        &expert_weights,
        &w_gate,
        &w_up,
        &w_down,
        DeepSeekRoutedMoeConfig {
            hidden_size: 3,
            intermediate_size: 2,
            top_k: 2,
            swiglu_limit: Some(1.0),
        },
        &mut output,
    )
    .unwrap();

    let expert_1_gate_0 = 0.5 * input[0] + 0.2 * input[1] - 0.1 * input[2];
    let expert_1_up_0 = 1.5 * input[0] - 0.3 * input[1] + 0.1 * input[2];
    let expert_1_gate_1 = -input[0] + 0.4 * input[1] + 0.3 * input[2];
    let expert_1_up_1 = 0.7 * input[0] + 0.6 * input[1] - 0.4 * input[2];
    let expert_1_hidden = [
        deepseek_swiglu(expert_1_gate_0, expert_1_up_0, Some(1.0)),
        deepseek_swiglu(expert_1_gate_1, expert_1_up_1, Some(1.0)),
    ];

    let expert_0_gate_0 = input[0] - 0.5 * input[1] + 0.25 * input[2];
    let expert_0_up_0 = -0.2 * input[0] + 0.4 * input[1] + 1.1 * input[2];
    let expert_0_gate_1 = -0.25 * input[0] + 0.75 * input[1] + 1.25 * input[2];
    let expert_0_up_1 = 0.8 * input[0] - 0.6 * input[1] + 0.2 * input[2];
    let expert_0_hidden = [
        deepseek_swiglu(expert_0_gate_0, expert_0_up_0, Some(1.0)),
        deepseek_swiglu(expert_0_gate_1, expert_0_up_1, Some(1.0)),
    ];

    let expected = [
        0.75 * (expert_1_hidden[0] * -0.7 + expert_1_hidden[1] * 0.6)
            + 0.25 * (expert_0_hidden[0] * 0.3 + expert_0_hidden[1] * -0.2),
        0.75 * (expert_1_hidden[0] * -0.1 + expert_1_hidden[1] * 0.25)
            + 0.25 * (expert_0_hidden[0] * 0.4 + expert_0_hidden[1] * 0.1),
        0.75 * (expert_1_hidden[0] * 0.35 + expert_1_hidden[1] * -0.45)
            + 0.25 * (expert_0_hidden[0] * -0.5 + expert_0_hidden[1] * 0.2),
    ];

    for (actual, expected) in output.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "actual={actual} expected={expected}"
        );
    }
}

#[test]
fn deepseek_routed_moe_rejects_bad_shapes() {
    let mut output = [0.0; 2];
    assert!(
        deepseek_routed_moe_forward(
            &[1.0, 2.0],
            &[0],
            &[1.0],
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0],
            DeepSeekRoutedMoeConfig {
                hidden_size: 2,
                intermediate_size: 2,
                top_k: 1,
                swiglu_limit: None,
            },
            &mut output,
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
