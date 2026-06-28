use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::request::{
    CUDA_HF_DECODE_SEQUENCE_DTYPE_F16, CudaHfDecodeSequenceRequest,
};
use crate::decode::hf_sequence::weight_plan::{
    CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT, CUDA_HF_WEIGHT_STRATEGY_GPU_STAGED,
    CudaHfDecodeSequenceWeightBlock, CudaHfDecodeSequenceWeightPlan, hash_weight_blocks,
};
use crate::smoke::status::SmokeStatus;

#[test]
fn hf_decode_sequence_accepts_qk_norm_descriptor_blocks() {
    let _guard = super::cuda_test_lock();

    let one = 0x3c00;
    let zero = 0x0000;
    let neg_one = 0xbc00;
    let embeddings = [one, zero, zero, one, neg_one, zero, zero, neg_one];
    let rms = [one, one];
    let matrix = [zero; 4];
    let lm_head = [zero, neg_one, one, zero, zero, one, neg_one, zero];
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &matrix,
        w_k: &matrix,
        q_norm_weight: Some(&rms),
        k_norm_weight: Some(&rms),
        w_v: &matrix,
        w_o: &matrix,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &matrix,
        w_up: &matrix,
        w_down: &matrix,
    };
    let blocks = weight_blocks(&embeddings, &rms, &matrix, &lm_head);
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
        embeddings: &embeddings,
        layers: &[layer],
        final_norm_weight: &rms,
        lm_head: &lm_head,
        weight_plan: Some(plan(&blocks)),
        weight_blocks: &blocks,
    }
    .run();

    if summary.status != SmokeStatus::Ok {
        return;
    }
    assert_eq!(summary.tokens, vec![1, 2, 3, 0]);
    assert_eq!(summary.planned_weight_blocks, 14);
    assert_eq!(summary.planned_weight_bytes, 108);
    assert_eq!(
        summary.planned_weight_descriptor_hash,
        hash_weight_blocks(&blocks)
    );
}

fn plan(blocks: &[CudaHfDecodeSequenceWeightBlock]) -> CudaHfDecodeSequenceWeightPlan {
    CudaHfDecodeSequenceWeightPlan {
        blocks: 14,
        gpu_resident_blocks: 7,
        gpu_staged_blocks: 7,
        weight_bytes: 108,
        gpu_resident_weight_bytes: 56,
        gpu_staged_weight_bytes: 52,
        descriptor_hash: hash_weight_blocks(blocks),
    }
}

fn weight_blocks(
    embeddings: &[u16],
    rms: &[u16],
    matrix: &[u16],
    lm_head: &[u16],
) -> Vec<CudaHfDecodeSequenceWeightBlock> {
    let bytes = [16, 4, 8, 4, 8, 4, 8, 8, 4, 8, 8, 8, 4, 16];
    let sources = [
        embeddings, rms, matrix, rms, matrix, rms, matrix, matrix, rms, matrix, matrix, matrix,
        rms, lm_head,
    ];
    let mut offset_bytes = 0;
    bytes
        .iter()
        .zip(sources)
        .enumerate()
        .map(|(index, (bytes, source))| {
            let strategy = if index < 7 {
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
