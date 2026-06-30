use crate::deepseek_moe::forward::{CudaDeepSeekMoeForwardInput, deepseek_moe_forward};
use crate::deepseek_moe::prepare::{CudaDeepSeekMegaMoePrepareInput, deepseek_megamoe_prepare};
use crate::deepseek_moe::probe::deepseek_moe_smoke;
use crate::deepseek_moe::summary::CudaDeepSeekMoeSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn deepseek_moe_summary_serializes_selected_expert_output() {
    let summary = CudaDeepSeekMoeSummary {
        status: SmokeStatus::Ok,
        hidden_size: 3,
        intermediate_size: 2,
        num_experts: 2,
        top_k: 2,
        swiglu_limit: 1.0,
        expert_ids: [1, 0],
        expert_weights: [0.75, 0.25],
        output: [0.1, -0.2, 0.3],
        output_hash: 44,
        mismatches: 0,
        max_abs_diff: 0.0,
        device_arena_bytes: 12,
        pinned_host_bytes: 12,
        d2h_bytes: 12,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"hidden_size\":3"));
    assert!(json.contains("\"expert_ids\":[1,0]"));
    assert!(json.contains("\"swiglu_limit\":1"));
    assert!(json.contains("\"mismatches\":0"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn deepseek_moe_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_moe_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_moe_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.hidden_size, 3);
    assert_eq!(second.intermediate_size, 2);
    assert_eq!(second.num_experts, 2);
    assert_eq!(second.top_k, 2);
    assert_eq!(second.expert_ids, [1, 0]);
    assert_eq!(second.mismatches, 0);
    assert!(second.max_abs_diff <= 1e-6);
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
    assert_eq!(second.output_hash, first.output_hash);
}

#[test]
fn deepseek_moe_forward_api_matches_reference() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let input = [1.2, -0.7, 0.3];
    let expert_ids = [1, 0];
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

    let summary = deepseek_moe_forward(CudaDeepSeekMoeForwardInput {
        hidden_size: 3,
        intermediate_size: 2,
        num_experts: 2,
        top_k: 2,
        clamp_swiglu: true,
        swiglu_limit: 1.0,
        input: &input,
        expert_ids: &expert_ids,
        expert_weights: &expert_weights,
        w_gate: &w_gate,
        w_up: &w_up,
        w_down: &w_down,
    });
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected = reference_moe_forward(
        3,
        2,
        2,
        true,
        1.0,
        &input,
        &expert_ids,
        &expert_weights,
        &w_gate,
        &w_up,
        &w_down,
    );
    assert_eq!(summary.hidden_size, 3);
    assert_eq!(summary.intermediate_size, 2);
    assert_eq!(summary.num_experts, 2);
    assert_eq!(summary.top_k, 2);
    assert!(summary.clamp_swiglu);
    for (actual, expected) in summary.output.iter().zip(expected.iter()) {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "actual={actual} expected={expected}"
        );
    }
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.output_hash != 0);
}

#[test]
fn deepseek_megamoe_prepare_matches_vllm_input_staging_contract() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let num_tokens = 2usize;
    let hidden_size = 128usize;
    let top_k = 3usize;
    let mut hidden_states = vec![0.0f32; num_tokens * hidden_size];
    for hidden in 0..hidden_size {
        hidden_states[hidden] = ((hidden % 11) as f32 - 5.0) * 0.125;
        hidden_states[hidden_size + hidden] = ((hidden % 7) as f32 - 3.0) * -0.25;
    }
    let topk_ids = [5i64, 2, 1, 7, 4, 3];
    let topk_weights = [0.5f32, 0.25, 0.125, 0.75, 0.125, 0.0625];
    let is_padding = [0u8, 1u8];

    let summary = deepseek_megamoe_prepare(CudaDeepSeekMegaMoePrepareInput {
        num_tokens: num_tokens as u32,
        hidden_size: hidden_size as u32,
        top_k: top_k as u32,
        hidden_states: &hidden_states,
        topk_ids: &topk_ids,
        topk_weights: &topk_weights,
        is_padding: Some(&is_padding),
    });
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected = reference_megamoe_prepare(
        &hidden_states,
        &topk_ids,
        &topk_weights,
        Some(&is_padding),
        num_tokens,
        hidden_size,
        top_k,
    );
    assert_eq!(summary.hidden_blocks, 1);
    assert_eq!(
        summary.x_fp8, expected.0,
        "MegaMoE fp8 hidden staging must match vLLM scale+cast contract"
    );
    assert_eq!(
        summary.x_scales, expected.1,
        "MegaMoE packed E8M0 hidden scales must match vLLM layout"
    );
    assert_eq!(
        summary.topk_ids, expected.2,
        "MegaMoE top-k IDs must repack to int64 and honor padding"
    );
    assert_eq!(
        summary.topk_weights, expected.3,
        "MegaMoE top-k weights must repack to f32 and honor padding"
    );
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.x_fp8_hash != 0);
    assert!(summary.x_scales_hash != 0);
    assert!(summary.topk_hash != 0);
}

#[allow(clippy::too_many_arguments)]
fn reference_moe_forward(
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    clamp_swiglu: bool,
    swiglu_limit: f32,
    input: &[f32],
    expert_ids: &[u32],
    expert_weights: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; hidden_size];
    for (rank, expert) in expert_ids.iter().copied().enumerate() {
        let expert = expert as usize;
        assert!(expert < num_experts);
        let expert_base = expert * intermediate_size * hidden_size;
        let down_base = expert * hidden_size * intermediate_size;
        let mut activation = vec![0.0f32; intermediate_size];
        for row in 0..intermediate_size {
            let start = expert_base + row * hidden_size;
            let gate = dot(&w_gate[start..start + hidden_size], input);
            let up = dot(&w_up[start..start + hidden_size], input);
            activation[row] = swiglu(gate, up, clamp_swiglu, swiglu_limit);
        }
        for hidden in 0..hidden_size {
            let start = down_base + hidden * intermediate_size;
            output[hidden] +=
                expert_weights[rank] * dot(&w_down[start..start + intermediate_size], &activation);
        }
    }
    output
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn swiglu(gate: f32, up: f32, clamp_swiglu: bool, swiglu_limit: f32) -> f32 {
    let gate = if clamp_swiglu {
        gate.min(swiglu_limit)
    } else {
        gate
    };
    let up = if clamp_swiglu {
        up.clamp(-swiglu_limit, swiglu_limit)
    } else {
        up
    };
    gate / (1.0 + (-gate).exp()) * up
}

#[allow(clippy::too_many_arguments)]
fn reference_megamoe_prepare(
    hidden_states: &[f32],
    topk_ids: &[i64],
    topk_weights: &[f32],
    is_padding: Option<&[u8]>,
    num_tokens: usize,
    hidden_size: usize,
    top_k: usize,
) -> (Vec<u8>, Vec<u32>, Vec<i64>, Vec<f32>) {
    let hidden_blocks = hidden_size.div_ceil(128);
    let mut x_fp8 = vec![0u8; num_tokens * hidden_size];
    let mut x_scales = vec![0u32; num_tokens * hidden_blocks];
    let mut topk_ids_out = vec![0i64; num_tokens * top_k];
    let mut topk_weights_out = vec![0.0f32; num_tokens * top_k];

    for token in 0..num_tokens {
        for block in 0..hidden_blocks {
            let mut packed = 0u32;
            for group in 0..4usize {
                let start = block * 128 + group * 32;
                let mut amax = 0.0f32;
                for offset in 0..32usize {
                    let hidden = start + offset;
                    let value = if hidden < hidden_size {
                        hidden_states[token * hidden_size + hidden]
                    } else {
                        0.0
                    };
                    amax = amax.max(value.abs());
                }
                amax = amax.max(1.0e-4);
                let scale_exp = ceil_e8m0_exponent(amax / 448.0);
                let scale = f32::from_bits((scale_exp as u32) << 23);
                packed |= (scale_exp as u32) << (group * 8);
                for offset in 0..32usize {
                    let hidden = start + offset;
                    if hidden < hidden_size {
                        let value = hidden_states[token * hidden_size + hidden] / scale;
                        x_fp8[token * hidden_size + hidden] = f32_to_f8_e4m3fn_nearest(value);
                    }
                }
            }
            x_scales[token * hidden_blocks + block] = packed;
        }

        let padding = match is_padding {
            Some(padding) => padding[token] != 0,
            None => false,
        };
        for rank in 0..top_k {
            let route = token * top_k + rank;
            topk_ids_out[route] = if padding { -1 } else { topk_ids[route] };
            topk_weights_out[route] = if padding { 0.0 } else { topk_weights[route] };
        }
    }

    (x_fp8, x_scales, topk_ids_out, topk_weights_out)
}

fn ceil_e8m0_exponent(scale: f32) -> u8 {
    let bits = scale.to_bits();
    let mut exp = ((bits >> 23) & 0xff) as u8;
    if bits & 0x7f_ffff != 0 {
        exp = exp.saturating_add(1);
    }
    exp.clamp(1, 254)
}

fn f32_to_f8_e4m3fn_nearest(value: f32) -> u8 {
    if !value.is_finite() {
        return 0x7f;
    }
    let mut best_bits = 0u8;
    let mut best_diff = f32::INFINITY;
    for bits in 0u8..=u8::MAX {
        let candidate = f8_e4m3fn_bits_to_f32(bits);
        if !candidate.is_finite() {
            continue;
        }
        let diff = (candidate - value).abs();
        if diff < best_diff {
            best_diff = diff;
            best_bits = bits;
        }
    }
    best_bits
}

fn f8_e4m3fn_bits_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 3) & 0x0f;
    let frac = bits & 0x07;
    if exp == 0 {
        if frac == 0 {
            return sign * 0.0;
        }
        return sign * ((frac as f32) * 0.125) * 2.0f32.powi(-6);
    }
    if exp == 0x0f && frac == 0x07 {
        return f32::NAN;
    }
    sign * (1.0 + (frac as f32) * 0.125) * 2.0f32.powi(exp as i32 - 7)
}
