use core::ptr;

use crate::decode::hf_chain::ffi::NervaCudaHfDecodeChainLayer;
use crate::decode::hf_chain::layer::{CUDA_HF_ATTENTION_FULL, CUDA_HF_ATTENTION_LINEAR_GDN};
use crate::decode::hf_sequence::ffi::{
    NervaCudaHfDecodeSamplerConfig, NervaCudaHfDecodeSequenceRequest,
    NervaCudaHfDecodeSequenceResult, run_hf_decode_sequence_u16,
};
use crate::decode::hf_sequence::request::CUDA_HF_DECODE_SEQUENCE_DTYPE_F16;
use crate::decode::hf_sequence::weight_plan::{
    CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT, CUDA_HF_WEIGHT_STRATEGY_GPU_STAGED,
    CudaHfDecodeSequenceWeightBlock, hash_weight_blocks,
};
use crate::smoke::status::SmokeStatus;

pub(super) struct TinyDescriptorWeights {
    pub embeddings: [u16; 8],
    pub rms: [u16; 2],
    pub matrix: [u16; 4],
    pub lm_head: [u16; 8],
}

impl TinyDescriptorWeights {
    pub(super) fn blocks(&self) -> Vec<CudaHfDecodeSequenceWeightBlock> {
        sequence_weight_blocks(&self.embeddings, &self.rms, &self.matrix, &self.lm_head)
    }
}

pub(super) fn tiny_descriptor_weights() -> TinyDescriptorWeights {
    let one = 0x3c00;
    let zero = 0x0000;
    let neg_one = 0xbc00;
    TinyDescriptorWeights {
        embeddings: [one, zero, zero, one, neg_one, zero, zero, neg_one],
        rms: [one, one],
        matrix: [zero; 4],
        lm_head: [zero, neg_one, one, zero, zero, one, neg_one, zero],
    }
}

pub(super) fn run_null_legacy_descriptor_decode(
    sampler: NervaCudaHfDecodeSamplerConfig,
) -> Option<(NervaCudaHfDecodeSequenceResult, [u32; 4])> {
    if crate::smoke::probe::smoke().status != SmokeStatus::Ok {
        return None;
    }
    let weights = tiny_descriptor_weights();
    let weight_blocks = weights.blocks();
    let layer = null_layer();
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
        sampler,
    };
    let mut out = NervaCudaHfDecodeSequenceResult::default();
    assert_eq!(run_hf_decode_sequence_u16(&request, &mut out), 0);
    Some((out, output_tokens))
}

#[test]
fn raw_descriptor_decode_rejects_linear_attention_kind() {
    let weights = tiny_descriptor_weights();
    let weight_blocks = weights.blocks();
    let mut layer = null_layer();
    layer.attention_kind = CUDA_HF_ATTENTION_LINEAR_GDN;
    let layers = [layer];
    let prompt_tokens = [0u32];
    let mut output_tokens = [0u32; 1];
    let request = NervaCudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        vocab_size: 4,
        layer_count: 1,
        steps: 1,
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
        output_token_capacity: 1,
        sampler: NervaCudaHfDecodeSamplerConfig::default(),
    };
    let mut out = NervaCudaHfDecodeSequenceResult::default();

    assert_eq!(run_hf_decode_sequence_u16(&request, &mut out), -1);
    assert_eq!(out.device_count, 0);
    assert_eq!(out.observed_tokens, 0);
}

fn null_layer() -> NervaCudaHfDecodeChainLayer {
    NervaCudaHfDecodeChainLayer {
        rms_attn_weight: ptr::null(),
        rms_mlp_weight: ptr::null(),
        w_q: ptr::null(),
        w_q_gate: ptr::null(),
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
        w_router: ptr::null(),
        w_expert_gate_up: ptr::null(),
        w_expert_down: ptr::null(),
        w_shared_expert_gate: ptr::null(),
        w_shared_expert_up: ptr::null(),
        w_shared_expert_down: ptr::null(),
        w_shared_expert_router: ptr::null(),
        linear_key_heads: 0,
        linear_value_heads: 0,
        linear_key_head_dim: 0,
        linear_value_head_dim: 0,
        linear_conv_kernel: 0,
        w_linear_conv: ptr::null(),
        w_linear_qkv: ptr::null(),
        w_linear_z: ptr::null(),
        w_linear_b: ptr::null(),
        w_linear_a: ptr::null(),
        w_linear_dt_bias: ptr::null(),
        w_linear_a_log: ptr::null(),
        w_linear_norm: ptr::null(),
        w_linear_out: ptr::null(),
        mlp_kind: 0,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: 0,
        attention_kind: CUDA_HF_ATTENTION_FULL,
        deepseek_mode: 0,
        deepseek_flags: 0,
        deepseek_hc_mult: 0,
        deepseek_hc_sinkhorn_iters: 0,
        deepseek_q_lora_rank: 0,
        deepseek_kv_lora_rank: 0,
        deepseek_o_lora_rank: 0,
        deepseek_o_groups: 0,
        deepseek_qk_nope_head_dim: 0,
        deepseek_qk_rope_head_dim: 0,
        deepseek_v_head_dim: 0,
        deepseek_compress_ratio: 0,
        deepseek_index_topk: 0,
        deepseek_index_n_heads: 0,
        deepseek_index_head_dim: 0,
        deepseek_router_num_groups: 0,
        deepseek_router_topk_groups: 0,
        deepseek_routed_scaling_factor: 1.0,
        deepseek_hc_eps: 0.0,
        deepseek_hc_post_alpha: 0.0,
    }
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
