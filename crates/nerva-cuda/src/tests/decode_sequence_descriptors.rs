use core::ptr;

use crate::decode::hf_chain::ffi::NervaCudaHfDecodeChainLayer;
use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::ffi::{
    run_hf_decode_sequence_u16, NervaCudaHfDecodeSequenceRequest, NervaCudaHfDecodeSequenceResult,
};
use crate::decode::hf_sequence::request::{
    CudaHfDecodeSequenceRequest, CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
};
use crate::decode::hf_sequence::weight_plan::{
    hash_weight_blocks, CudaHfDecodeSequenceWeightBlock, CudaHfDecodeSequenceWeightPlan,
    CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT, CUDA_HF_WEIGHT_STRATEGY_GPU_STAGED,
};
use crate::smoke::status::SmokeStatus;

#[test]
fn declared_weight_descriptors_override_legacy_weight_pointers() {
    let _guard = super::cuda_test_lock();

    let one = 0x3c00;
    let zero = 0x0000;
    let neg_one = 0xbc00;
    let embeddings = [one, zero, zero, one, neg_one, zero, zero, neg_one];
    let rms = [one, one];
    let matrix = [zero; 4];
    let lm_head = [zero, neg_one, one, zero, zero, one, neg_one, zero];
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
    let weight_blocks = sequence_weight_blocks(&embeddings, &rms, &matrix, &lm_head);
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
    let _guard = super::cuda_test_lock();

    if crate::smoke::probe::smoke().status != SmokeStatus::Ok {
        return;
    }
    let one = 0x3c00;
    let zero = 0x0000;
    let neg_one = 0xbc00;
    let embeddings = [one, zero, zero, one, neg_one, zero, zero, neg_one];
    let rms = [one, one];
    let matrix = [zero; 4];
    let lm_head = [zero, neg_one, one, zero, zero, one, neg_one, zero];
    let weight_blocks = sequence_weight_blocks(&embeddings, &rms, &matrix, &lm_head);
    let layer = NervaCudaHfDecodeChainLayer {
        rms_attn_weight: ptr::null(),
        rms_mlp_weight: ptr::null(),
        w_q: ptr::null(),
        w_k: ptr::null(),
        q_norm_weight: ptr::null(),
        k_norm_weight: ptr::null(),
        w_v: ptr::null(),
        w_o: ptr::null(),
        q_bias: ptr::null(),
        k_bias: ptr::null(),
        v_bias: ptr::null(),
        o_bias: ptr::null(),
        w_gate: ptr::null(),
        w_up: ptr::null(),
        w_down: ptr::null(),
    };
    let layers = [layer];
    let prompt_tokens = [0u32];
    let mut output_tokens = [0u32; 4];
    let request = NervaCudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        vocab_size: 4,
        layer_count: 1,
        steps: 4,
        seed_token: 0,
        prompt_tokens: prompt_tokens.as_ptr(),
        prompt_token_count: 1,
        has_eos_token: 0,
        eos_token: 0,
        rms_eps: 1e-5,
        rope_theta: 0.0,
        embeddings: ptr::null(),
        layers: layers.as_ptr(),
        final_norm_weight: ptr::null(),
        lm_head: ptr::null(),
        planned_weight_blocks: 12,
        planned_gpu_resident_blocks: 6,
        planned_gpu_staged_blocks: 6,
        planned_weight_bytes: 100,
        planned_gpu_resident_weight_bytes: 52,
        planned_gpu_staged_weight_bytes: 48,
        planned_weight_descriptors: weight_blocks.as_ptr(),
        planned_weight_descriptor_count: 12,
        planned_weight_descriptor_hash: hash_weight_blocks(&weight_blocks),
        output_tokens: output_tokens.as_mut_ptr(),
        output_token_capacity: 4,
    };
    let mut out = NervaCudaHfDecodeSequenceResult::default();
    let return_code = run_hf_decode_sequence_u16(&request, &mut out);
    assert_eq!(return_code, 0);
    assert_eq!(out.status, 0);
    assert_eq!(out.descriptor_gpu_resident_h2d_bytes, 52);
    assert_eq!(out.descriptor_gpu_staged_h2d_bytes, 48);
    assert_eq!(output_tokens, [1, 2, 3, 0]);
}

fn sequence_weight_blocks(
    embeddings: &[u16],
    rms: &[u16],
    matrix: &[u16],
    lm_head: &[u16],
) -> Vec<CudaHfDecodeSequenceWeightBlock> {
    let bytes = [16, 4, 8, 8, 8, 8, 4, 8, 8, 8, 4, 16];
    let sources = [
        embeddings, rms, matrix, matrix, matrix, matrix, rms, matrix, matrix, matrix, rms, lm_head,
    ];
    let mut offset_bytes = 0;
    bytes
        .iter()
        .zip(sources)
        .enumerate()
        .map(|(index, (bytes, source))| {
            let strategy = if index < 6 {
                CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT
            } else {
                CUDA_HF_WEIGHT_STRATEGY_GPU_STAGED
            };
            let block = CudaHfDecodeSequenceWeightBlock {
                host_source: source.as_ptr(),
                block_id: index as u64 + 1,
                block_version: 0,
                offset_bytes,
                bytes: *bytes,
                strategy,
                reserved: 0,
                ..CudaHfDecodeSequenceWeightBlock::default()
            };
            offset_bytes += *bytes;
            block
        })
        .collect()
}
