use crate::attention::block::KvAttentionBlock;
use crate::attention::exact::mla::{
    DeepSeekMlaDecodeScratch, DeepSeekMlaDecodeShape, DeepSeekMlaPrefillScratch,
    exact_deepseek_mla_decode_mqa_into, exact_deepseek_mla_prefill_causal_mqa_into,
};
use crate::attention::exact::run::exact_blockwise_attention_into;
use crate::attention::scratch::BlockwiseAttentionScratch;
use crate::attention::smoke::{BlockwiseAttentionSmokeStatus, blockwise_attention_smoke};
use crate::common::shape::TransformerBlockShape;
use crate::tests::support::dense_attention_reference;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::ledger::TokenLedger;

#[test]
fn blockwise_attention_matches_dense_reference_across_tiers() {
    let shape = TransformerBlockShape::new(4, 2, 4);
    let query = [0.5, -1.0, 0.25, 0.75];
    let keys = [0.1, 0.2, 0.3, 0.4, 0.0, -0.5, 0.6, 0.2, 0.7, 0.1, -0.2, 0.3];
    let values = [
        1.0, 0.0, 0.5, -0.5, -1.0, 2.0, 0.25, 0.75, 0.3, -0.8, 1.5, 0.2,
    ];
    let blocks = [
        KvAttentionBlock::new(&keys[..4], &values[..4], 1, MemoryTier::Dram),
        KvAttentionBlock::new(&keys[4..], &values[4..], 2, MemoryTier::Vram),
    ];
    let mut scratch = BlockwiseAttentionScratch::new(shape).unwrap();
    let mut output = [0.0; 4];
    let mut ledger = TokenLedger::new(11);

    exact_blockwise_attention_into(
        shape,
        &query,
        &blocks,
        &mut scratch,
        &mut output,
        &mut ledger,
    )
    .unwrap();

    let expected = dense_attention_reference(shape, &query, &keys, &values, 3);
    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(ledger.event_count(LedgerEventKind::CpuActivity), 1);
    assert_eq!(ledger.event_count(LedgerEventKind::DeviceActivity), 1);
    assert_eq!(ledger.total_latency_ns(), 3);
    assert_eq!(ledger.hot_path_allocations, 0);
}

#[test]
fn blockwise_attention_maps_grouped_query_heads_to_compact_kv() {
    let shape = TransformerBlockShape::new_with_kv_heads(8, 4, 2, 8);
    let query = [0.5, -0.25, 0.1, 0.8, -0.3, 0.4, 0.9, -0.2];
    let keys = [
        0.2, 0.1, -0.4, 0.3, 0.7, -0.5, 0.25, 0.6, -0.1, 0.9, 0.4, -0.8,
    ];
    let values = [
        1.0, -0.5, 0.25, 0.75, -1.0, 0.5, 0.8, -0.2, 0.6, 1.1, -0.4, 0.3,
    ];
    let block = [KvAttentionBlock::new(&keys, &values, 3, MemoryTier::Dram)];
    let mut scratch = BlockwiseAttentionScratch::new(shape).unwrap();
    let mut output = [0.0; 8];
    let mut ledger = TokenLedger::new(12);

    exact_blockwise_attention_into(
        shape,
        &query,
        &block,
        &mut scratch,
        &mut output,
        &mut ledger,
    )
    .unwrap();

    let expected = dense_attention_reference(shape, &query, &keys, &values, 3);
    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(ledger.event_count(LedgerEventKind::CpuActivity), 1);
    assert_eq!(ledger.events[0].bytes, 96);
}

#[test]
fn blockwise_attention_rejects_empty_and_malformed_blocks() {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let query = [1.0, 0.0];
    let mut scratch = BlockwiseAttentionScratch::new(shape).unwrap();
    let mut output = [0.0; 2];
    let mut ledger = TokenLedger::new(0);

    assert!(
        exact_blockwise_attention_into(shape, &query, &[], &mut scratch, &mut output, &mut ledger)
            .is_err()
    );

    let bad_block = [KvAttentionBlock::new(
        &[1.0],
        &[1.0, 0.0],
        1,
        MemoryTier::Dram,
    )];
    assert!(
        exact_blockwise_attention_into(
            shape,
            &query,
            &bad_block,
            &mut scratch,
            &mut output,
            &mut ledger,
        )
        .is_err()
    );
}

#[test]
fn deepseek_mla_decode_mqa_matches_expanded_mha_reference() {
    let shape = DeepSeekMlaDecodeShape::new(2, 3, 2, 1, 2);
    let q_nope = [0.2, -0.3, 0.4, 0.1];
    let q_pe = [0.15, -0.25];
    let kv_c = [
        0.3, -0.1, 0.2, //
        -0.4, 0.5, 0.1, //
        0.2, 0.4, -0.3,
    ];
    let k_pe = [0.05, -0.2, 0.3];
    let w_uk_lnp = [
        0.3, -0.2, 0.1, 0.4, //
        -0.5, 0.2, 0.6, -0.1, //
        0.7, 0.3, -0.2, 0.5,
    ];
    let w_uv_lnv = [
        0.2, -0.4, 0.5, 0.1, //
        -0.3, 0.6, 0.4, -0.2, //
        0.7, 0.2, -0.1, 0.3,
    ];
    let softmax_scale = 0.7;
    let mut scratch = DeepSeekMlaDecodeScratch::new(shape).unwrap();
    let mut output = [0.0; 4];

    exact_deepseek_mla_decode_mqa_into(
        shape,
        &q_nope,
        &q_pe,
        &kv_c,
        &k_pe,
        &w_uk_lnp,
        &w_uv_lnv,
        softmax_scale,
        &mut scratch,
        &mut output,
    )
    .unwrap();

    let expected = expanded_mla_mha_reference(
        shape,
        &q_nope,
        &q_pe,
        &kv_c,
        &k_pe,
        &w_uk_lnp,
        &w_uv_lnv,
        softmax_scale,
    );
    for (actual, expected) in output.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
}

#[test]
fn deepseek_mla_prefill_causal_mqa_matches_repeated_decode_reference() {
    let shape = DeepSeekMlaDecodeShape::new(2, 3, 2, 1, 2);
    let q_nope = [
        0.2, -0.3, 0.4, 0.1, //
        0.1, 0.5, -0.2, 0.3, //
        -0.4, 0.25, 0.6, -0.1,
    ];
    let q_pe = [
        0.15, -0.25, //
        0.05, 0.35, //
        -0.1, 0.2,
    ];
    let kv_c = [
        0.3, -0.1, 0.2, //
        -0.4, 0.5, 0.1, //
        0.2, 0.4, -0.3,
    ];
    let k_pe = [0.05, -0.2, 0.3];
    let w_uk_lnp = [
        0.3, -0.2, 0.1, 0.4, //
        -0.5, 0.2, 0.6, -0.1, //
        0.7, 0.3, -0.2, 0.5,
    ];
    let w_uv_lnv = [
        0.2, -0.4, 0.5, 0.1, //
        -0.3, 0.6, 0.4, -0.2, //
        0.7, 0.2, -0.1, 0.3,
    ];
    let softmax_scale = 0.7;
    let tokens = 3;
    let mut prefill_scratch = DeepSeekMlaPrefillScratch::new(shape, tokens).unwrap();
    let mut prefill_output = [0.0; 12];

    exact_deepseek_mla_prefill_causal_mqa_into(
        shape,
        tokens,
        &q_nope,
        &q_pe,
        &kv_c,
        &k_pe,
        &w_uk_lnp,
        &w_uv_lnv,
        softmax_scale,
        &mut prefill_scratch,
        &mut prefill_output,
    )
    .unwrap();

    let mut decode_scratch = DeepSeekMlaDecodeScratch::new(shape).unwrap();
    let mut decode_output = [0.0; 4];
    for token in 0..tokens {
        exact_deepseek_mla_decode_mqa_into(
            shape,
            &q_nope[token * shape.q_nope_len().unwrap()..][..shape.q_nope_len().unwrap()],
            &q_pe[token * shape.q_pe_len().unwrap()..][..shape.q_pe_len().unwrap()],
            &kv_c[..(token + 1) * shape.kv_lora_rank],
            &k_pe[..token + 1],
            &w_uk_lnp,
            &w_uv_lnv,
            softmax_scale,
            &mut decode_scratch,
            &mut decode_output,
        )
        .unwrap();
        let actual =
            &prefill_output[token * shape.output_len().unwrap()..][..shape.output_len().unwrap()];
        for (actual, expected) in actual.iter().zip(decode_output.iter()) {
            assert!(
                (actual - expected).abs() < 1e-6,
                "token={token} actual={actual} expected={expected}"
            );
        }
    }
}

#[test]
fn deepseek_mla_decode_shape_covers_v3_and_v4_profiles() {
    let v3 = DeepSeekMlaDecodeShape::new(128, 512, 128, 64, 128);
    v3.validate().unwrap();
    assert_eq!(v3.q_nope_len().unwrap(), 16_384);
    assert_eq!(v3.q_pe_len().unwrap(), 8_192);
    assert_eq!(v3.w_uk_len().unwrap(), 8_388_608);
    assert_eq!(v3.w_uv_len().unwrap(), 8_388_608);

    let v4 = DeepSeekMlaDecodeShape::new(64, 512, 448, 64, 512);
    v4.validate().unwrap();
    assert_eq!(v4.q_nope_len().unwrap(), 28_672);
    assert_eq!(v4.q_pe_len().unwrap(), 4_096);
    assert_eq!(v4.w_uk_len().unwrap(), 14_680_064);
    assert_eq!(v4.w_uv_len().unwrap(), 16_777_216);
}

#[test]
fn deepseek_mla_prefill_rejects_bad_token_shapes() {
    let shape = DeepSeekMlaDecodeShape::new(1, 2, 1, 1, 1);
    let mut scratch = DeepSeekMlaPrefillScratch::new(shape, 1).unwrap();
    let mut output = [0.0; 1];

    assert!(
        exact_deepseek_mla_prefill_causal_mqa_into(
            shape,
            2,
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[1.0, 0.0, 0.5, 0.25],
            &[1.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            1.0,
            &mut scratch,
            &mut output,
        )
        .is_err()
    );

    assert!(
        exact_deepseek_mla_prefill_causal_mqa_into(
            shape,
            1,
            &[1.0],
            &[1.0],
            &[1.0],
            &[1.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            1.0,
            &mut scratch,
            &mut output,
        )
        .is_err()
    );
}

#[test]
fn deepseek_mla_decode_rejects_bad_shapes() {
    let shape = DeepSeekMlaDecodeShape::new(1, 2, 1, 1, 1);
    let mut scratch = DeepSeekMlaDecodeScratch::new(shape).unwrap();
    let mut output = [0.0; 1];

    assert!(
        exact_deepseek_mla_decode_mqa_into(
            shape,
            &[1.0],
            &[1.0],
            &[],
            &[],
            &[1.0, 1.0],
            &[1.0, 1.0],
            1.0,
            &mut scratch,
            &mut output,
        )
        .is_err()
    );

    assert!(
        exact_deepseek_mla_decode_mqa_into(
            shape,
            &[1.0],
            &[1.0],
            &[1.0, 0.0],
            &[1.0],
            &[1.0],
            &[1.0, 1.0],
            1.0,
            &mut scratch,
            &mut output,
        )
        .is_err()
    );

    let mut other_scratch =
        DeepSeekMlaDecodeScratch::new(DeepSeekMlaDecodeShape::new(1, 3, 1, 1, 1)).unwrap();
    assert!(
        exact_deepseek_mla_decode_mqa_into(
            shape,
            &[1.0],
            &[1.0],
            &[1.0, 0.0],
            &[1.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            1.0,
            &mut other_scratch,
            &mut output,
        )
        .is_err()
    );
}

#[test]
fn blockwise_attention_smoke_reports_tier_events() {
    let summary = blockwise_attention_smoke().unwrap();
    assert_eq!(summary.status, BlockwiseAttentionSmokeStatus::Ok);
    assert_eq!(summary.hidden, 2);
    assert_eq!(summary.heads, 1);
    assert_eq!(summary.blocks, 2);
    assert_eq!(summary.tokens, 4);
    assert_eq!(summary.cpu_block_events, 1);
    assert_eq!(summary.device_block_events, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"device_block_events\":1"));
}

fn expanded_mla_mha_reference(
    shape: DeepSeekMlaDecodeShape,
    q_nope: &[f32],
    q_pe: &[f32],
    kv_c: &[f32],
    k_pe: &[f32],
    w_uk_lnp: &[f32],
    w_uv_lnv: &[f32],
    softmax_scale: f32,
) -> Vec<f32> {
    let tokens = kv_c.len() / shape.kv_lora_rank;
    let mut output = vec![0.0; shape.output_len().unwrap()];
    for head in 0..shape.heads {
        let q_nope = &q_nope[head * shape.qk_nope_head_dim..][..shape.qk_nope_head_dim];
        let q_pe = &q_pe[head * shape.qk_rope_head_dim..][..shape.qk_rope_head_dim];
        let mut scores = vec![0.0; tokens];
        let mut values = vec![0.0; tokens * shape.v_head_dim];

        for token in 0..tokens {
            let kv = &kv_c[token * shape.kv_lora_rank..][..shape.kv_lora_rank];
            let k_pe = &k_pe[token * shape.qk_rope_head_dim..][..shape.qk_rope_head_dim];
            let mut score = 0.0f32;
            for p in 0..shape.qk_nope_head_dim {
                let mut k_nope = 0.0f32;
                for (latent, kv_value) in kv.iter().copied().enumerate() {
                    let weight_index = (latent * shape.heads + head) * shape.qk_nope_head_dim + p;
                    k_nope += kv_value * w_uk_lnp[weight_index];
                }
                score += q_nope[p] * k_nope;
            }
            for r in 0..shape.qk_rope_head_dim {
                score += q_pe[r] * k_pe[r];
            }
            scores[token] = score * softmax_scale;

            for v in 0..shape.v_head_dim {
                let mut value = 0.0f32;
                for (latent, kv_value) in kv.iter().copied().enumerate() {
                    let weight_index = (latent * shape.heads + head) * shape.v_head_dim + v;
                    value += kv_value * w_uv_lnv[weight_index];
                }
                values[token * shape.v_head_dim + v] = value;
            }
        }

        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let normalizer = scores
            .iter()
            .map(|score| (*score - max_score).exp())
            .sum::<f32>();
        for token in 0..tokens {
            let prob = (scores[token] - max_score).exp() / normalizer;
            for v in 0..shape.v_head_dim {
                output[head * shape.v_head_dim + v] += prob * values[token * shape.v_head_dim + v];
            }
        }
    }
    output
}
