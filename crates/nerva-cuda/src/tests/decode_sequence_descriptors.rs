use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::ffi::NervaCudaHfDecodeSamplerConfig;
use crate::decode::hf_sequence::request::{
    CUDA_HF_DECODE_SEQUENCE_DTYPE_F16, CudaHfDecodeSamplerConfig, CudaHfDecodeSequenceRequest,
};
use crate::decode::hf_sequence::weight_plan::{CudaHfDecodeSequenceWeightPlan, hash_weight_blocks};
use crate::smoke::status::SmokeStatus;

use super::decode_sequence_descriptor_blocks::{
    run_null_legacy_descriptor_decode, tiny_descriptor_weights,
};

#[test]
fn declared_weight_descriptors_override_legacy_weight_pointers() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let weights = tiny_descriptor_weights();
    let zero = 0x0000;
    let one = 0x3c00;
    let poisoned_embeddings = [zero; 8];
    let poisoned_rms = [zero; 2];
    let poisoned_matrix = [one; 4];
    let poisoned_lm_head = [zero; 8];
    let poisoned_layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &poisoned_rms,
        rms_mlp_weight: &poisoned_rms,
        w_q: &poisoned_matrix,
        w_k: &poisoned_matrix,
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &poisoned_matrix,
        w_o: &poisoned_matrix,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &poisoned_matrix,
        w_up: &poisoned_matrix,
        w_down: &poisoned_matrix,
    };
    let poisoned_layers = [poisoned_layer];
    let weight_blocks = weights.blocks();
    let summary = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        vocab_size: 4,
        steps: 4,
        seed_token: 0,
        prompt_tokens: &[0],
        eos_token: None,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &poisoned_embeddings,
        layers: &poisoned_layers,
        final_norm_weight: &poisoned_rms,
        lm_head: &poisoned_lm_head,
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 12,
            gpu_resident_blocks: 6,
            gpu_staged_blocks: 6,
            weight_bytes: 100,
            gpu_resident_weight_bytes: 52,
            gpu_staged_weight_bytes: 48,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    }
    .run();

    if summary.status != SmokeStatus::Ok {
        assert_eq!(summary.status, SmokeStatus::Unavailable);
        return;
    }
    assert_eq!(summary.tokens, vec![1, 2, 3, 0]);
    assert_eq!(summary.descriptor_gpu_resident_h2d_bytes, 52);
    assert_eq!(summary.descriptor_gpu_staged_h2d_bytes, 48);
    assert_eq!(summary.planned_weight_descriptor_count, 12);
}

#[test]
fn declared_weight_descriptors_accept_null_legacy_weight_pointers() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let sampler = NervaCudaHfDecodeSamplerConfig {
        temperature: 0.0,
        ..NervaCudaHfDecodeSamplerConfig::default()
    };
    let Some((out, output_tokens)) = run_null_legacy_descriptor_decode(sampler) else {
        return;
    };
    assert_eq!(out.status, 0);
    assert_eq!(out.descriptor_gpu_resident_h2d_bytes, 52);
    assert_eq!(out.descriptor_gpu_staged_h2d_bytes, 48);
    assert_eq!(output_tokens, [1, 2, 3, 0]);
}

#[test]
fn declared_weight_descriptors_support_temperature_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let sampler = NervaCudaHfDecodeSamplerConfig {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        reserved: 0,
        seed: 0,
    };
    let Some((out, output_tokens)) = run_null_legacy_descriptor_decode(sampler) else {
        return;
    };
    assert_eq!(out.status, 0);
    assert_eq!(out.descriptor_gpu_resident_h2d_bytes, 52);
    assert_eq!(out.descriptor_gpu_staged_h2d_bytes, 48);
    assert_eq!(out.observed_tokens, 4);
    assert_eq!(output_tokens, [0, 0, 1, 1]);
}
