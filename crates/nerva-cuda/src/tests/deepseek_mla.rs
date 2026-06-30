use crate::deepseek_mla::decode::{CudaDeepSeekMlaDecodeInput, deepseek_mla_decode};
use crate::deepseek_mla::probe::deepseek_mla_smoke;
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
