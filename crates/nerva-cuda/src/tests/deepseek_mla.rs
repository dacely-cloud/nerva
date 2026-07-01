use crate::deepseek_mla::decode::{CudaDeepSeekMlaDecodeInput, deepseek_mla_decode};
use crate::deepseek_mla::probe::{deepseek_mla_smoke, deepseek_qkv_rmsnorm_smoke};
use crate::deepseek_mla::qkv_norm::{
    CudaDeepSeekQKvRmsNormSummary, deepseek_qkv_rmsnorm, deepseek_qkv_rmsnorm_reference,
};
use crate::deepseek_mla::summary::CudaDeepSeekMlaSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn deepseek_mla_summary_serializes_shape_and_output() {
    let summary = CudaDeepSeekMlaSummary {
        status: SmokeStatus::Ok,
        heads: 2,
        tokens: 3,
        kv_lora_rank: 3,
        qk_nope_head_dim: 2,
        qk_rope_head_dim: 1,
        v_head_dim: 2,
        softmax_scale: 0.7,
        output: [0.1, -0.2, 0.3, -0.4],
        output_hash: 11,
        mismatches: 0,
        max_abs_diff: 0.0,
        device_arena_bytes: 16,
        pinned_host_bytes: 16,
        d2h_bytes: 16,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"heads\":2"));
    assert!(json.contains("\"tokens\":3"));
    assert!(json.contains("\"kv_lora_rank\":3"));
    assert!(json.contains("\"qk_nope_head_dim\":2"));
    assert!(json.contains("\"qk_rope_head_dim\":1"));
    assert!(json.contains("\"v_head_dim\":2"));
    assert!(json.contains("\"output\":[0.1,-0.2,0.3,-0.4]"));
    assert!(json.contains("\"mismatches\":0"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn deepseek_qkv_rmsnorm_summary_serializes_outputs() {
    let summary = CudaDeepSeekQKvRmsNormSummary {
        status: SmokeStatus::Ok,
        return_code: 0,
        cuda_error: 0,
        num_tokens: 2,
        q_size: 4,
        kv_size: 3,
        eps: 1e-5,
        q_out: vec![0.1, -0.2],
        kv_out: vec![0.3, -0.4],
        output_hash: 17,
        device_arena_bytes: 64,
        pinned_host_bytes: 32,
        h2d_bytes: 48,
        d2h_bytes: 32,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"num_tokens\":2"));
    assert!(json.contains("\"q_size\":4"));
    assert!(json.contains("\"kv_size\":3"));
    assert!(json.contains("\"q_out\":[0.1,-0.2]"));
    assert!(json.contains("\"kv_out\":[0.3,-0.4]"));
    assert!(json.contains("\"kernel_launches\":1"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn deepseek_qkv_rmsnorm_reference_matches_vllm_fused_q_kv_rmsnorm_math() {
    let q = [
        1.0, -2.0, 3.0, -4.0, // token 0
        -0.5, 1.5, -2.5, 3.5, // token 1
    ];
    let kv = [
        0.25, -0.75, 1.25, // token 0
        -1.5, 2.0, -2.5, // token 1
    ];
    let q_weight = [0.5, 1.0, -1.5, 2.0];
    let kv_weight = [1.25, -0.5, 0.75];
    let reference =
        deepseek_qkv_rmsnorm_reference(&q, &kv, &q_weight, &kv_weight, 2, 4, 3, 1e-5).unwrap();

    assert_eq!(reference.q_out.len(), 8);
    assert_eq!(reference.kv_out.len(), 6);
    let q_row0_variance = (1.0f32 + 4.0 + 9.0 + 16.0) / 4.0;
    let q_row0_rrms = 1.0 / (q_row0_variance + 1e-5).sqrt();
    assert_close(reference.q_out[0], 1.0 * q_row0_rrms * 0.5, 1e-6);
    assert_close(reference.q_out[3], -4.0 * q_row0_rrms * 2.0, 1e-6);
    let kv_row1_variance = (2.25f32 + 4.0 + 6.25) / 3.0;
    let kv_row1_rrms = 1.0 / (kv_row1_variance + 1e-5).sqrt();
    assert_close(reference.kv_out[3], -1.5 * kv_row1_rrms * 1.25, 1e-6);
    assert_close(reference.kv_out[5], -2.5 * kv_row1_rrms * 0.75, 1e-6);
    assert!(
        deepseek_qkv_rmsnorm_reference(&q, &kv, &q_weight[..3], &kv_weight, 2, 4, 3, 1e-5).is_err()
    );
}

#[test]
fn deepseek_qkv_rmsnorm_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_qkv_rmsnorm_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_qkv_rmsnorm_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.num_tokens, 2);
    assert_eq!(second.q_size, 4);
    assert_eq!(second.kv_size, 3);
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
    assert_eq!(second.output_hash, first.output_hash);
    assert_eq!(second.q_out, first.q_out);
    assert_eq!(second.kv_out, first.kv_out);
}

#[test]
fn deepseek_qkv_rmsnorm_api_matches_vllm_fused_q_kv_rmsnorm_math() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let q = [
        1.0, -2.0, 3.0, -4.0, // token 0
        -0.5, 1.5, -2.5, 3.5, // token 1
    ];
    let kv = [
        0.25, -0.75, 1.25, // token 0
        -1.5, 2.0, -2.5, // token 1
    ];
    let q_weight = [0.5, 1.0, -1.5, 2.0];
    let kv_weight = [1.25, -0.5, 0.75];
    let summary = deepseek_qkv_rmsnorm(&q, &kv, &q_weight, &kv_weight, 2, 4, 3, 1e-5);
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected =
        deepseek_qkv_rmsnorm_reference(&q, &kv, &q_weight, &kv_weight, 2, 4, 3, 1e-5).unwrap();
    assert_eq!(summary.num_tokens, 2);
    assert_eq!(summary.q_size, 4);
    assert_eq!(summary.kv_size, 3);
    for (actual, expected) in summary.q_out.iter().zip(expected.q_out.iter()) {
        assert_close(*actual, *expected, 1e-5);
    }
    for (actual, expected) in summary.kv_out.iter().zip(expected.kv_out.iter()) {
        assert_close(*actual, *expected, 1e-5);
    }
    assert_eq!(summary.kernel_launches, 1);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.output_hash != 0);
}

#[test]
fn deepseek_mla_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_mla_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_mla_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.heads, 2);
    assert_eq!(second.tokens, 3);
    assert_eq!(second.kv_lora_rank, 3);
    assert_eq!(second.qk_nope_head_dim, 2);
    assert_eq!(second.qk_rope_head_dim, 1);
    assert_eq!(second.v_head_dim, 2);
    assert_eq!(second.mismatches, 0);
    assert!(second.max_abs_diff <= 1e-6);
    assert!(second.output.iter().all(|value| value.is_finite()));
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
    assert_eq!(second.output_hash, first.output_hash);
}

#[test]
fn deepseek_mla_decode_api_matches_expanded_mha_reference() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let q_nope = [0.2, -0.3, 0.4, 0.1];
    let q_pe = [0.15, -0.25];
    let kv_c = [0.3, -0.1, 0.2, -0.4, 0.5, 0.1, 0.2, 0.4, -0.3];
    let k_pe = [0.05, -0.2, 0.3];
    let w_uk = [
        0.3, -0.2, 0.1, 0.4, -0.5, 0.2, 0.6, -0.1, 0.7, 0.3, -0.2, 0.5,
    ];
    let w_uv = [
        0.2, -0.4, 0.5, 0.1, -0.3, 0.6, 0.4, -0.2, 0.7, 0.2, -0.1, 0.3,
    ];

    let summary = deepseek_mla_decode(CudaDeepSeekMlaDecodeInput {
        heads: 2,
        tokens: 3,
        kv_lora_rank: 3,
        qk_nope_head_dim: 2,
        qk_rope_head_dim: 1,
        v_head_dim: 2,
        softmax_scale: 0.7,
        q_nope: &q_nope,
        q_pe: &q_pe,
        kv_c: &kv_c,
        k_pe: &k_pe,
        w_uk: &w_uk,
        w_uv: &w_uv,
    });
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected = reference_mla_decode(
        2, 3, 3, 2, 1, 2, 0.7, &q_nope, &q_pe, &kv_c, &k_pe, &w_uk, &w_uv,
    );
    assert_eq!(summary.heads, 2);
    assert_eq!(summary.tokens, 3);
    assert_eq!(summary.kv_lora_rank, 3);
    assert_eq!(summary.qk_nope_head_dim, 2);
    assert_eq!(summary.qk_rope_head_dim, 1);
    assert_eq!(summary.v_head_dim, 2);
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

#[allow(clippy::too_many_arguments)]
fn reference_mla_decode(
    heads: usize,
    tokens: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    softmax_scale: f32,
    q_nope: &[f32],
    q_pe: &[f32],
    kv_c: &[f32],
    k_pe: &[f32],
    w_uk: &[f32],
    w_uv: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; heads * v_head_dim];
    for head in 0..heads {
        let mut scores = vec![0.0f32; tokens];
        let mut max_score = f32::NEG_INFINITY;
        for token in 0..tokens {
            let kv = &kv_c[token * kv_lora_rank..(token + 1) * kv_lora_rank];
            let mut score = 0.0f32;
            for nope in 0..qk_nope_head_dim {
                let mut k_nope = 0.0f32;
                for latent in 0..kv_lora_rank {
                    let w_idx = (latent * heads + head) * qk_nope_head_dim + nope;
                    k_nope += kv[latent] * w_uk[w_idx];
                }
                score += q_nope[head * qk_nope_head_dim + nope] * k_nope;
            }
            for rope in 0..qk_rope_head_dim {
                score +=
                    q_pe[head * qk_rope_head_dim + rope] * k_pe[token * qk_rope_head_dim + rope];
            }
            scores[token] = score * softmax_scale;
            max_score = max_score.max(scores[token]);
        }
        let normalizer = scores
            .iter()
            .map(|score| (score - max_score).exp())
            .sum::<f32>();
        for token in 0..tokens {
            let prob = (scores[token] - max_score).exp() / normalizer;
            let kv = &kv_c[token * kv_lora_rank..(token + 1) * kv_lora_rank];
            for v in 0..v_head_dim {
                let mut value = 0.0f32;
                for latent in 0..kv_lora_rank {
                    let w_idx = (latent * heads + head) * v_head_dim + v;
                    value += kv[latent] * w_uv[w_idx];
                }
                output[head * v_head_dim + v] += prob * value;
            }
        }
    }
    output
}

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual} expected={expected} tolerance={tolerance}"
    );
}
