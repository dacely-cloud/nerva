use crate::decode::hf_chain::layer::{
    CUDA_HF_ATTENTION_DEEPSEEK_MLA, CUDA_HF_ATTENTION_LINEAR_GDN, CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR,
    CUDA_HF_DEEPSEEK_FLAG_HASH_ROUTER, CUDA_HF_DEEPSEEK_FLAG_MOE,
    CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS, CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER,
    CUDA_HF_DEEPSEEK_MODE_V3_MLA, CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED,
    CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER, CUDA_HF_DEEPSEEK_MODE_V4_SWA,
    CUDA_HF_DEEPSEEK_MODE_V32_MLA_INDEXER, CUDA_HF_MLP_DENSE, CUDA_HF_MLP_SPARSE_MOE,
    CudaHfDecodeChainLayer, CudaHfDeepSeekLayer, CudaHfLinearGdnLayer,
};
use crate::decode::hf_sequence::footprint::estimate_sequence_footprint;
use crate::decode::hf_sequence::layout_plan::{
    CUDA_HF_SEQUENCE_MISSING_OFFSET, CudaHfDecodeSequenceLayoutPlanRequest,
};
use crate::decode::hf_sequence::request::{
    CUDA_HF_DECODE_SEQUENCE_DTYPE_F16, CudaHfDecodeSamplerConfig, CudaHfDecodeSequenceRequest,
};
use crate::decode::hf_sequence::session::request::{
    CudaHfDecodeSequenceExperimentalRtConfig, CudaHfDecodeSequenceSessionConfig,
};
use crate::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use crate::decode::hf_sequence::weight_plan::{
    CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT, CudaHfDecodeSequenceWeightBlock,
    CudaHfDecodeSequenceWeightPlan, hash_weight_blocks,
};
use crate::smoke::status::SmokeStatus;

use super::decode_sequence_descriptor_blocks::{
    run_null_legacy_descriptor_decode, tiny_descriptor_weights,
};

#[test]
fn linear_gdn_layer_validation_preserves_layout_metadata() {
    let hidden = 4;
    let rms = vec![0x3c00; hidden];
    let router = vec![0x3c00; 4 * hidden];
    let expert_gate_up = vec![0x3c00; 4 * 2 * 3 * hidden];
    let expert_down = vec![0x3c00; 4 * hidden * 3];
    let linear_conv = vec![0x3c00; 28];
    let linear_qkv = vec![0x3c00; 28];
    let linear_z = vec![0x3c00; 12];
    let linear_b = vec![0x3c00; 4];
    let linear_a = vec![0x3c00; 4];
    let linear_dt_bias = vec![0x3c00; 1];
    let linear_a_log = vec![0.0f32];
    let linear_norm = vec![0x0000, 0x3f80, 0x0000, 0x3f80, 0x0000, 0x3f80];
    let linear_out = vec![0x3c00; 12];
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &[],
        w_q_gate: None,
        w_k: &[],
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &[],
        w_o: &[],
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &[],
        w_up: &[],
        w_down: &[],
        w_router: Some(&router),
        w_expert_gate_up: Some(&expert_gate_up),
        w_expert_down: Some(&expert_down),
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: Some(CudaHfLinearGdnLayer {
            key_heads: 1,
            value_heads: 1,
            key_head_dim: 2,
            value_head_dim: 3,
            conv_kernel: 4,
            w_conv: &linear_conv,
            w_qkv: &linear_qkv,
            w_z: &linear_z,
            w_b: &linear_b,
            w_a: &linear_a,
            dt_bias: &linear_dt_bias,
            a_log: &linear_a_log,
            norm_weight: &linear_norm,
            w_out: &linear_out,
        }),
        deepseek: None,
        mlp_kind: CUDA_HF_MLP_SPARSE_MOE,
        moe_intermediate: 3,
        shared_expert_intermediate: 0,
        num_experts: 4,
        experts_per_token: 2,
        norm_topk_prob: true,
        attention_kind: CUDA_HF_ATTENTION_LINEAR_GDN,
    };

    assert_eq!(layer.validate(hidden, 4, 4, 2, 8), None);
    let ffi = layer.to_ffi();
    assert_eq!(ffi.linear_key_heads, 1);
    assert_eq!(ffi.linear_value_head_dim, 3);
    assert!(!ffi.w_linear_a_log.is_null());
}

#[test]
fn deepseek_mla_layer_validation_preserves_layout_metadata() {
    let hidden = 4096;
    let rms = vec![0x3c00; hidden];
    let deepseek = CudaHfDeepSeekLayer {
        mode: CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER,
        flags: CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER
            | CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR
            | CUDA_HF_DEEPSEEK_FLAG_HASH_ROUTER
            | CUDA_HF_DEEPSEEK_FLAG_MOE,
        hc_mult: 4,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        o_lora_rank: 1536,
        o_groups: 8,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
        compress_ratio: 4,
        index_topk: 2048,
        index_n_heads: 64,
        index_head_dim: 128,
        router_num_groups: 0,
        router_topk_groups: 0,
        routed_scaling_factor: 1.0,
    };
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &[],
        w_q_gate: None,
        w_k: &[],
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &[],
        w_o: &[],
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &[],
        w_up: &[],
        w_down: &[],
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: Some(deepseek),
        mlp_kind: CUDA_HF_MLP_SPARSE_MOE,
        moe_intermediate: 2048,
        shared_expert_intermediate: 0,
        num_experts: 128,
        experts_per_token: 8,
        norm_topk_prob: true,
        attention_kind: CUDA_HF_ATTENTION_DEEPSEEK_MLA,
    };

    assert_eq!(layer.validate(hidden, hidden, 512, 128, 4096), None);

    let ffi = layer.to_ffi();
    assert_eq!(ffi.attention_kind, CUDA_HF_ATTENTION_DEEPSEEK_MLA);
    assert_eq!(ffi.deepseek_mode, deepseek.mode);
    assert_eq!(ffi.deepseek_flags, deepseek.flags);
    assert_eq!(ffi.deepseek_hc_mult, deepseek.hc_mult as u32);
    assert_eq!(ffi.deepseek_q_lora_rank, deepseek.q_lora_rank as u32);
    assert_eq!(ffi.deepseek_kv_lora_rank, deepseek.kv_lora_rank as u32);
    assert_eq!(ffi.deepseek_o_lora_rank, deepseek.o_lora_rank as u32);
    assert_eq!(ffi.deepseek_o_groups, deepseek.o_groups as u32);
    assert_eq!(
        ffi.deepseek_qk_nope_head_dim,
        deepseek.qk_nope_head_dim as u32
    );
    assert_eq!(
        ffi.deepseek_qk_rope_head_dim,
        deepseek.qk_rope_head_dim as u32
    );
    assert_eq!(ffi.deepseek_v_head_dim, deepseek.v_head_dim as u32);
    assert_eq!(ffi.deepseek_compress_ratio, deepseek.compress_ratio as u32);
    assert_eq!(ffi.deepseek_index_topk, deepseek.index_topk as u32);
    assert_eq!(ffi.deepseek_index_n_heads, deepseek.index_n_heads as u32);
    assert_eq!(ffi.deepseek_index_head_dim, deepseek.index_head_dim as u32);
    assert_eq!(
        ffi.deepseek_router_num_groups,
        deepseek.router_num_groups as u32
    );
    assert_eq!(
        ffi.deepseek_router_topk_groups,
        deepseek.router_topk_groups as u32
    );
    assert_eq!(
        ffi.deepseek_routed_scaling_factor,
        deepseek.routed_scaling_factor
    );

    let descriptor = layer.to_descriptor_layout_ffi();
    assert!(descriptor.w_q.is_null());
    assert!(descriptor.w_gate.is_null());
    assert_eq!(descriptor.deepseek_mode, deepseek.mode);
    assert_eq!(descriptor.deepseek_flags, deepseek.flags);
    assert_eq!(
        descriptor.deepseek_router_num_groups,
        deepseek.router_num_groups as u32
    );
}

#[test]
fn deepseek_v3_mla_shape_matches_vllm_contract() {
    let deepseek = CudaHfDeepSeekLayer {
        mode: CUDA_HF_DEEPSEEK_MODE_V3_MLA,
        flags: 0,
        hc_mult: 0,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        o_lora_rank: 0,
        o_groups: 0,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
        compress_ratio: 1,
        index_topk: 0,
        index_n_heads: 0,
        index_head_dim: 0,
        router_num_groups: 8,
        router_topk_groups: 4,
        routed_scaling_factor: 2.5,
    };

    assert!(deepseek.is_v3_mla());
    assert!(!deepseek.is_v4_mla());
    assert_eq!(deepseek.qk_head_dim(), Some(192));

    let shape = deepseek
        .v3_mla_shape(128)
        .expect("DeepSeek V3 dimensions should form an MLA shape");
    assert_eq!(shape.num_heads, 128);
    assert_eq!(shape.qk_head_dim, 192);
    assert_eq!(shape.q_rows, 24_576);
    assert_eq!(shape.kv_cache_width, 576);
    assert_eq!(shape.kv_b_rows, 32_768);
    assert_eq!(shape.value_rows, 16_384);
}

#[test]
fn deepseek_v4_mla_shape_does_not_reuse_v3_cache_contract() {
    let deepseek = tiny_deepseek_v4_descriptor_layer()
        .deepseek
        .expect("fixture should carry DeepSeek metadata");

    assert!(deepseek.is_v4_mla());
    assert!(!deepseek.is_v3_mla());
    assert_eq!(deepseek.v3_mla_shape(2), None);
}

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
        w_q_gate: None,
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
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: None,
        mlp_kind: 0,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: false,
        attention_kind: crate::decode::hf_chain::layer::CUDA_HF_ATTENTION_FULL,
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

    assert_raw_descriptor_decode_matches_request(CudaHfDecodeSamplerConfig::greedy());
}

#[test]
fn declared_weight_descriptors_support_temperature_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    assert_raw_descriptor_decode_matches_request(CudaHfDecodeSamplerConfig::vllm_default());
}

#[test]
fn declared_sparse_moe_descriptor_footprint_uses_router_and_experts() {
    let sparse_layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_q_gate: None,
        w_k: &[],
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &[],
        w_o: &[],
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &[],
        w_up: &[],
        w_down: &[],
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: None,
        mlp_kind: CUDA_HF_MLP_SPARSE_MOE,
        moe_intermediate: 2,
        shared_expert_intermediate: 0,
        num_experts: 3,
        experts_per_token: 2,
        norm_topk_prob: true,
        attention_kind: crate::decode::hf_chain::layer::CUDA_HF_ATTENTION_FULL,
    };
    let layers = [sparse_layer];
    let request = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 1,
        kv_heads: 1,
        head_dim: 4,
        intermediate: 8,
        vocab_size: 8,
        steps: 4,
        seed_token: 0,
        prompt_tokens: &[0],
        eos_token: None,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: 448,
            gpu_resident_weight_bytes: 448,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: 1,
        }),
        weight_blocks: &[],
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    };

    let footprint = estimate_sequence_footprint(&request).unwrap();

    assert_eq!(footprint.resident_weight_bytes, 448);
    assert_eq!(footprint.layout_bytes, 584);
}

#[test]
fn declared_deepseek_v4_descriptor_footprint_counts_storage_widths_and_hc_blocks() {
    let deepseek_layer = tiny_deepseek_v4_descriptor_layer();
    let layers = [deepseek_layer];
    let request = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        steps: 2,
        seed_token: 0,
        prompt_tokens: &[0],
        eos_token: None,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: 1306,
            gpu_resident_weight_bytes: 1306,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: 1,
        }),
        weight_blocks: &[],
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    };

    let footprint = estimate_sequence_footprint(&request).unwrap();

    assert_eq!(footprint.resident_weight_bytes, 1306);
    assert_eq!(footprint.layout_bytes, 584);
}

#[test]
fn declared_deepseek_v4_descriptor_run_reaches_native_execution_guard() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let deepseek_layer = tiny_deepseek_v4_descriptor_layer();
    let layers = [deepseek_layer];
    let weight_storage = vec![0u16; 1306 / 2];
    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: 1306,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let request = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        steps: 2,
        seed_token: 0,
        prompt_tokens: &[0],
        eos_token: None,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: 1306,
            gpu_resident_weight_bytes: 1306,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    };

    let summary = request.run();
    if summary.status == SmokeStatus::Unavailable {
        return;
    }

    assert_eq!(summary.status, SmokeStatus::Failed);
    assert_eq!(summary.planned_footprint.resident_weight_bytes, 1306);
    assert_eq!(summary.planned_weight_descriptor_count, 1);
    assert_eq!(
        summary.planned_weight_descriptor_hash,
        hash_weight_blocks(&weight_blocks)
    );
    assert!(
        summary
            .error
            .as_deref()
            .is_some_and(|error| error.contains("cuda_error=801")),
        "expected cudaErrorNotSupported guard, got {:?}",
        summary.error
    );
}

#[test]
fn deepseek_v32_layout_plan_names_projection_and_indexer_offsets() {
    let layer = tiny_deepseek_v32_descriptor_layer();
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V3.2 descriptor layer");

    assert_eq!(plan.attention_kind, CUDA_HF_ATTENTION_DEEPSEEK_MLA);
    assert_eq!(plan.deepseek_mode, CUDA_HF_DEEPSEEK_MODE_V32_MLA_INDEXER);
    assert_eq!(plan.deepseek_qk_head_dim, 2);
    assert_eq!(plan.deepseek_q_rows, 4);
    assert_eq!(plan.deepseek_kv_cache_width, 3);
    assert_eq!(plan.deepseek_kv_b_rows, 4);
    assert_eq!(plan.deepseek_value_rows, 2);
    assert_eq!(plan.rms_attn, 40);
    assert_eq!(plan.w_q, 48);
    assert_eq!(plan.deepseek_q_a_scale, 52);
    assert_eq!(plan.q_norm, 54);
    assert_eq!(plan.deepseek_q_b, 58);
    assert_eq!(plan.deepseek_q_b_scale, 62);
    assert_eq!(plan.w_k, 64);
    assert_eq!(plan.deepseek_kv_a_scale, 70);
    assert_eq!(plan.k_norm, 72);
    assert_eq!(plan.w_v, 76);
    assert_eq!(plan.deepseek_kv_b_scale, 80);
    assert_eq!(plan.w_o, 82);
    assert_eq!(plan.deepseek_o_a_scale, 86);
    assert_eq!(plan.deepseek_indexer_q, 88);
    assert_eq!(plan.deepseek_indexer_q_scale, 92);
    assert_eq!(plan.deepseek_indexer_k, 94);
    assert_eq!(plan.deepseek_indexer_k_scale, 98);
    assert_eq!(plan.deepseek_indexer_k_norm, 100);
    assert_eq!(plan.deepseek_indexer_k_norm_bias, 104);
    assert_eq!(plan.deepseek_indexer_weights, 108);
    assert_eq!(plan.rms_mlp, 116);
    assert_eq!(plan.deepseek_o_b, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_eq!(
        plan.deepseek_compressor_ape,
        CUDA_HF_SEQUENCE_MISSING_OFFSET
    );
    assert_eq!(plan.layout_bytes, 584);
    assert!(plan.resident_weight_bytes > 0);

    let request = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        steps: 2,
        seed_token: 0,
        prompt_tokens: &[0],
        eos_token: None,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: 1,
        }),
        weight_blocks: &[],
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    };
    let footprint = estimate_sequence_footprint(&request).unwrap();

    assert_eq!(footprint.resident_weight_bytes, plan.resident_weight_bytes);
    assert_eq!(footprint.resident_kv_bytes, 192);
    assert_eq!(footprint.scratch_bytes, 200);
}

#[test]
fn deepseek_v32_dense_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v32_descriptor_layer();
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V3.2 descriptor layer");
    let weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        max_context_tokens: 4,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: Default::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }

    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3.2 DeepSeek should pass session creation: {:?}",
        created.summary.error
    );
    let mut session = created.session.expect("V3.2 session handle should exist");
    assert_eq!(
        created.summary.resident_weight_bytes,
        plan.resident_weight_bytes
    );

    let summary = session.run(&[0], 2, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V3.2 DeepSeek dense path should run through sampling: {:?}",
        summary.error
    );
    assert_eq!(summary.steps, 2);
    assert_eq!(summary.tokens.len(), 2);
    assert_eq!(summary.kv_tokens, 2);
    assert_eq!(summary.graph_replays, 2);
    assert_eq!(summary.deepseek_v3_grouped_router_selections, 0);
    assert_eq!(summary.deepseek_v4_bias_router_selections, 0);
    assert_eq!(summary.deepseek_v4_hash_router_selections, 0);
    assert!(summary.graph_nodes > 0);
}

#[test]
fn deepseek_v32_sparse_moe_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut layer = tiny_deepseek_v32_descriptor_layer();
    layer.mlp_kind = CUDA_HF_MLP_SPARSE_MOE;
    layer.moe_intermediate = 4;
    layer.shared_expert_intermediate = 2;
    layer.num_experts = 2;
    layer.experts_per_token = 1;
    layer.norm_topk_prob = true;
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.flags |= CUDA_HF_DEEPSEEK_FLAG_MOE | CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS;
        deepseek.router_num_groups = 1;
        deepseek.router_topk_groups = 1;
        deepseek.routed_scaling_factor = 1.0;
        deepseek
    });
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V3.2 sparse MoE layer");
    assert_ne!(plan.w_router, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.w_expert_gate_up, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.w_expert_down, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        max_context_tokens: 4,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: Default::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }

    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3.2 DeepSeek sparse MoE should pass session creation: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V3.2 sparse MoE session handle should exist");

    let summary = session.run(&[0], 2, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V3.2 DeepSeek sparse MoE path should run through sampling: {:?}",
        summary.error
    );
    assert_eq!(summary.steps, 2);
    assert_eq!(summary.tokens.len(), 2);
    assert_eq!(summary.kv_tokens, 2);
    assert_eq!(summary.graph_replays, 2);
    assert_eq!(
        summary.deepseek_v3_grouped_router_selections,
        summary.graph_replays
    );
    assert_eq!(summary.deepseek_v4_bias_router_selections, 0);
    assert_eq!(summary.deepseek_v4_hash_router_selections, 0);
    assert!(summary.graph_nodes > 0);
}

#[test]
fn deepseek_v4_swa_dense_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v4_swa_dense_descriptor_layer();
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V4 SWA dense layer");
    assert_ne!(
        plan.deepseek_attention_sink,
        CUDA_HF_SEQUENCE_MISSING_OFFSET
    );
    assert_ne!(plan.deepseek_o_b, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        max_context_tokens: 4,
        rms_eps: 1e-5,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: Default::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }

    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V4 SWA dense DeepSeek should pass session creation: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V4 SWA dense session handle should exist");

    let summary = session.run(&[0], 2, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V4 SWA dense DeepSeek path should run through sampling: {:?}",
        summary.error
    );
    assert_eq!(summary.steps, 2);
    assert_eq!(summary.tokens.len(), 2);
    assert_eq!(summary.kv_tokens, 2);
    assert_eq!(summary.graph_replays, 2);
    assert_eq!(summary.deepseek_v3_grouped_router_selections, 0);
    assert_eq!(summary.deepseek_v4_bias_router_selections, 0);
    assert_eq!(summary.deepseek_v4_hash_router_selections, 0);
    assert!(summary.graph_nodes > 0);
}

#[test]
fn deepseek_v4_swa_sparse_moe_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut layer = tiny_deepseek_v4_swa_dense_descriptor_layer();
    layer.mlp_kind = CUDA_HF_MLP_SPARSE_MOE;
    layer.moe_intermediate = 4;
    layer.shared_expert_intermediate = 2;
    layer.num_experts = 2;
    layer.experts_per_token = 1;
    layer.norm_topk_prob = true;
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.flags |= CUDA_HF_DEEPSEEK_FLAG_MOE | CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS;
        deepseek.routed_scaling_factor = 1.0;
        deepseek
    });
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V4 SWA sparse MoE layer");
    assert_ne!(plan.w_router, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.w_expert_gate_up, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.w_expert_down, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        max_context_tokens: 4,
        rms_eps: 1e-5,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: Default::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }

    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V4 SWA sparse MoE DeepSeek should pass session creation: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V4 SWA sparse MoE session handle should exist");

    let summary = session.run(&[0], 2, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V4 SWA sparse MoE DeepSeek path should run through sampling: {:?}",
        summary.error
    );
    assert_eq!(summary.steps, 2);
    assert_eq!(summary.tokens.len(), 2);
    assert_eq!(summary.kv_tokens, 2);
    assert_eq!(summary.graph_replays, 2);
    assert_eq!(summary.deepseek_v3_grouped_router_selections, 0);
    assert_eq!(
        summary.deepseek_v4_bias_router_selections,
        summary.graph_replays
    );
    assert_eq!(summary.deepseek_v4_hash_router_selections, 0);
    assert!(summary.graph_nodes > 0);
}

#[test]
fn deepseek_v4_swa_hash_moe_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut layer = tiny_deepseek_v4_swa_dense_descriptor_layer();
    layer.mlp_kind = CUDA_HF_MLP_SPARSE_MOE;
    layer.moe_intermediate = 4;
    layer.shared_expert_intermediate = 2;
    layer.num_experts = 2;
    layer.experts_per_token = 1;
    layer.norm_topk_prob = true;
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.flags |= CUDA_HF_DEEPSEEK_FLAG_MOE
            | CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS
            | CUDA_HF_DEEPSEEK_FLAG_HASH_ROUTER;
        deepseek.routed_scaling_factor = 1.0;
        deepseek
    });
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V4 SWA hash MoE layer");
    assert_ne!(plan.w_router, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.w_expert_gate_up, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.w_expert_down, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        max_context_tokens: 4,
        rms_eps: 1e-5,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: Default::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }

    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V4 SWA hash MoE DeepSeek should pass session creation: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V4 SWA hash MoE session handle should exist");

    let summary = session.run(&[0], 2, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V4 SWA hash MoE DeepSeek path should run through sampling: {:?}",
        summary.error
    );
    assert_eq!(summary.steps, 2);
    assert_eq!(summary.tokens.len(), 2);
    assert_eq!(summary.kv_tokens, 2);
    assert_eq!(summary.graph_replays, 2);
    assert_eq!(summary.deepseek_v3_grouped_router_selections, 0);
    assert_eq!(summary.deepseek_v4_bias_router_selections, 0);
    assert_eq!(
        summary.deepseek_v4_hash_router_selections,
        summary.graph_replays
    );
    assert!(summary.graph_nodes > 0);
}

#[test]
fn deepseek_v4_compressed_dense_short_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut layer = tiny_deepseek_v4_swa_dense_descriptor_layer();
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.mode = CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED;
        deepseek.flags |= CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR;
        deepseek.compress_ratio = 4;
        deepseek
    });
    with_tiny_deepseek_v4_descriptor_session(layer, 8, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed dense DeepSeek should pass session creation: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed dense session handle should exist");

        let summary = session.run(&[0], 2, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "V4 compressed dense short context should run through SWA-only sampling: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 2);
        assert_eq!(summary.tokens.len(), 2);
        assert_eq!(summary.kv_tokens, 2);
        assert_eq!(summary.graph_replays, 2);
        assert!(summary.graph_nodes > 0);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 0);
        assert_eq!(summary.deepseek_indexer_state_writes, 0);
        assert_eq!(summary.deepseek_indexer_kv_writes, 0);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 0);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 0);
        assert_eq!(summary.deepseek_sparse_topk_selections, 0);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 0);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 0);
    });
}

#[test]
fn deepseek_v4_swa_dense_respects_sliding_window_limit() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut rt = CudaHfDecodeSequenceExperimentalRtConfig::default();
    rt.local_window_tokens = 3;
    with_tiny_deepseek_v4_descriptor_session_with_rt(
        tiny_deepseek_v4_swa_dense_descriptor_layer(),
        8,
        rt,
        |created| {
            if created.summary.status == SmokeStatus::Unavailable {
                return;
            }
            assert_eq!(
                created.summary.status,
                SmokeStatus::Ok,
                "V4 SWA session should create before sliding-window accounting: {:?}",
                created.summary.error
            );
            let mut session = created.session.expect("V4 SWA session handle should exist");

            let summary = session.run(&[0], 6, None);
            assert_eq!(
                summary.status,
                SmokeStatus::Ok,
                "V4 SWA decode should use the configured local window: {:?}",
                summary.error
            );
            assert_eq!(summary.steps, 6);
            assert_eq!(summary.kv_tokens, 6);
            assert_eq!(summary.graph_replays, 6);
            assert_eq!(summary.deepseek_raw_attention_tokens_scanned, 15);
            assert_eq!(summary.deepseek_compressed_kv_writes, 0);
            assert_eq!(summary.deepseek_compressed_kv_attention_reads, 0);
            assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 0);
        },
    );
}

#[test]
fn deepseek_v4_compressed_indexer_short_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v4_descriptor_layer();
    with_tiny_deepseek_v4_descriptor_session(layer, 8, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer DeepSeek should pass session creation: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed-indexer session handle should exist");

        let summary = session.run(&[0], 2, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer short context should run through SWA-only sampling: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 2);
        assert_eq!(summary.tokens.len(), 2);
        assert_eq!(summary.kv_tokens, 2);
        assert_eq!(summary.graph_replays, 2);
        assert!(summary.graph_nodes > 0);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 0);
        assert_eq!(summary.deepseek_indexer_state_writes, summary.graph_replays);
        assert_eq!(summary.deepseek_indexer_kv_writes, 0);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 0);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 0);
        assert_eq!(summary.deepseek_sparse_topk_selections, 0);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 0);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 0);
    });
}

#[test]
fn deepseek_v4_compressed_indexer_writes_first_boundary_cache() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v4_descriptor_layer();
    with_tiny_deepseek_v4_descriptor_session(layer, 8, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer DeepSeek should create before runtime boundary checks: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed-indexer session handle should exist");

        let summary = session.run(&[0], 4, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "context length at compress_ratio should write first compressed cache boundary: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 4);
        assert_eq!(summary.tokens.len(), 4);
        assert_eq!(summary.kv_tokens, 4);
        assert_eq!(summary.graph_replays, 4);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 1);
        assert_eq!(summary.deepseek_indexer_state_writes, summary.graph_replays);
        assert_eq!(summary.deepseek_indexer_kv_writes, 1);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 1);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 1);
        assert_eq!(summary.deepseek_sparse_topk_selections, 1);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 1);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 0);
    });
}

#[test]
fn deepseek_v4_compressed_indexer_writes_realistic_indexer_cache_width() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut layer = tiny_deepseek_v4_descriptor_layer();
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.index_head_dim = 128;
        deepseek
    });
    with_tiny_deepseek_v4_descriptor_session(layer, 8, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer realistic cache width should create: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed-indexer session handle should exist");

        let summary = session.run(&[0], 4, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "realistic indexer head width should write first compressed indexer cache boundary: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 4);
        assert_eq!(summary.kv_tokens, 4);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 1);
        assert_eq!(summary.deepseek_indexer_state_writes, summary.graph_replays);
        assert_eq!(summary.deepseek_indexer_kv_writes, 1);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 1);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 1);
        assert_eq!(summary.deepseek_sparse_topk_selections, 1);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 1);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 0);
    });
}

#[test]
fn deepseek_v4_compressed_indexer_runs_past_first_boundary_with_compressed_attention() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v4_descriptor_layer();
    with_tiny_deepseek_v4_descriptor_session(layer, 8, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer DeepSeek should create before runtime boundary checks: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed-indexer session handle should exist");

        let summary = session.run(&[0], 5, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "context past compress_ratio should read native compressed cache: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 5);
        assert_eq!(summary.tokens.len(), 5);
        assert_eq!(summary.kv_tokens, 5);
        assert_eq!(summary.graph_replays, 5);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 1);
        assert_eq!(summary.deepseek_indexer_state_writes, summary.graph_replays);
        assert_eq!(summary.deepseek_indexer_kv_writes, 1);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 2);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 2);
        assert_eq!(summary.deepseek_sparse_topk_selections, 2);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 2);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 0);
    });
}

#[test]
fn deepseek_v4_compressed_indexer_tracks_compressed_attention_scan_growth() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v4_descriptor_layer();
    with_tiny_deepseek_v4_descriptor_session(layer, 12, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer DeepSeek should create before scan accounting: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed-indexer session handle should exist");

        let summary = session.run(&[0], 8, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "context through the second compression boundary should scan compressed slots: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 8);
        assert_eq!(summary.tokens.len(), 8);
        assert_eq!(summary.kv_tokens, 8);
        assert_eq!(summary.graph_replays, 8);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 2);
        assert_eq!(summary.deepseek_indexer_state_writes, summary.graph_replays);
        assert_eq!(summary.deepseek_indexer_kv_writes, 2);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 5);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 6);
        assert_eq!(summary.deepseek_sparse_topk_selections, 5);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 6);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 0);
    });
}

#[test]
fn deepseek_v4_compressed_indexer_limits_attention_to_sparse_topk() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut layer = tiny_deepseek_v4_descriptor_layer();
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.index_topk = 1;
        deepseek.index_head_dim = 128;
        deepseek
    });
    with_tiny_deepseek_v4_descriptor_session(layer, 12, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer sparse top-k session should create: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed-indexer session handle should exist");

        let summary = session.run(&[0], 8, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "C4 sparse indexer should cap compressed attention slots: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 8);
        assert_eq!(summary.kv_tokens, 8);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 2);
        assert_eq!(summary.deepseek_indexer_state_writes, summary.graph_replays);
        assert_eq!(summary.deepseek_indexer_kv_writes, 2);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 5);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 5);
        assert_eq!(summary.deepseek_sparse_topk_selections, 5);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 5);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 2);
    });
}

#[test]
fn deepseek_v4_compressed_indexer_session_reserves_compressor_runtime_caches() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut swa_kv_bytes = None;
    with_tiny_deepseek_v4_descriptor_session(
        tiny_deepseek_v4_swa_dense_descriptor_layer(),
        8,
        |created| {
            if created.summary.status == SmokeStatus::Unavailable {
                return;
            }
            assert_eq!(
                created.summary.status,
                SmokeStatus::Ok,
                "V4 SWA baseline session should create: {:?}",
                created.summary.error
            );
            assert_eq!(
                created.summary.deepseek_v4_attention_aux_streams, 3,
                "V4 SWA attention sessions should provision aux GEMM streams"
            );
            assert_eq!(
                created.summary.deepseek_v4_attention_events, 4,
                "V4 SWA attention sessions should provision aux stream events"
            );
            assert_eq!(
                created.summary.deepseek_v4_swa_kv_bytes, 576,
                "V4 SWA must reserve one vLLM-aligned fp8_ds_mla page"
            );
            swa_kv_bytes = Some(created.summary.resident_kv_bytes);
        },
    );

    let Some(swa_kv_bytes) = swa_kv_bytes else {
        return;
    };

    with_tiny_deepseek_v4_descriptor_session(tiny_deepseek_v4_descriptor_layer(), 8, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer session should create: {:?}",
            created.summary.error
        );
        assert_eq!(
            created.summary.deepseek_v4_attention_aux_streams, 3,
            "V4 attention sessions should provision the vLLM-style aux GEMM streams"
        );
        assert_eq!(
            created.summary.deepseek_v4_attention_events, 4,
            "V4 attention sessions should provision fan-out/join events for aux streams"
        );
        assert_eq!(
            created.summary.deepseek_v4_swa_kv_bytes, 576,
            "V4 compressed-indexer layers still reserve the local SWA fp8_ds_mla page"
        );
        assert!(
            created.summary.resident_kv_bytes > swa_kv_bytes,
            "compressed-indexer V4 must reserve compressor/indexer runtime caches: {} <= {}",
            created.summary.resident_kv_bytes,
            swa_kv_bytes
        );
        assert_eq!(
            created.summary.resident_kv_bytes - swa_kv_bytes,
            512 + 576 + 512 + 576,
            "V4 C4 compressed/indexer runtime caches must reserve vLLM-aligned packed pages"
        );
    });

    let mut c128_layer = tiny_deepseek_v4_descriptor_layer();
    c128_layer.deepseek = c128_layer.deepseek.map(|mut deepseek| {
        deepseek.compress_ratio = 128;
        deepseek
    });
    with_tiny_deepseek_v4_descriptor_session(c128_layer, 256, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 C128 compressed-indexer session should create: {:?}",
            created.summary.error
        );
        assert_eq!(
            created.summary.deepseek_v4_attention_aux_streams, 3,
            "V4 C128 attention sessions should provision aux GEMM streams"
        );
        assert_eq!(
            created.summary.deepseek_v4_attention_events, 4,
            "V4 C128 attention sessions should provision aux stream events"
        );
        assert_eq!(
            created.summary.deepseek_v4_swa_kv_bytes, 2304,
            "V4 SWA cache must reserve four 64-token fp8_ds_mla pages for 256 tokens"
        );
        assert_eq!(
            created.summary.resident_kv_bytes - 2048 - created.summary.deepseek_v4_swa_kv_bytes,
            4096 + 576 + 4096 + 576,
            "V4 C128 compressed/indexer runtime caches must reserve two-token vLLM-aligned packed pages"
        );
    });
}

#[test]
fn deepseek_v4_layout_plan_names_compressor_and_indexer_offsets() {
    let layer = tiny_deepseek_v4_descriptor_layer();
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V4 descriptor layer");

    assert_eq!(plan.attention_kind, CUDA_HF_ATTENTION_DEEPSEEK_MLA);
    assert_eq!(
        plan.deepseek_mode,
        CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER
    );
    assert_eq!(plan.deepseek_qk_head_dim, 0);
    assert_eq!(plan.deepseek_q_rows, 0);
    assert_eq!(plan.deepseek_kv_cache_width, 0);
    assert_eq!(plan.deepseek_kv_b_rows, 0);
    assert_eq!(plan.deepseek_value_rows, 0);
    assert_eq!(plan.rms_attn, 78);
    assert_eq!(plan.deepseek_attention_sink, 382);
    assert_eq!(plan.w_q, 386);
    assert_eq!(plan.deepseek_q_a_scale, 390);
    assert_eq!(plan.deepseek_q_b, 391);
    assert_eq!(plan.deepseek_q_b_scale, 395);
    assert_eq!(plan.q_norm, 396);
    assert_eq!(plan.w_k, 398);
    assert_eq!(plan.deepseek_kv_a_scale, 402);
    assert_eq!(plan.k_norm, 403);
    assert_eq!(plan.w_o, 405);
    assert_eq!(plan.deepseek_o_a_scale, 409);
    assert_eq!(plan.deepseek_o_b, 410);
    assert_eq!(plan.deepseek_o_b_scale, 418);
    assert_eq!(plan.deepseek_compressor_ape, 419);
    assert_eq!(plan.deepseek_compressor_wkv, 451);
    assert_eq!(plan.deepseek_compressor_wgate, 467);
    assert_eq!(plan.deepseek_compressor_norm, 483);
    assert_eq!(plan.deepseek_indexer_q, 485);
    assert_eq!(plan.deepseek_indexer_q_scale, 487);
    assert_eq!(plan.deepseek_indexer_compressor_ape, 488);
    assert_eq!(plan.deepseek_indexer_compressor_wkv, 520);
    assert_eq!(plan.deepseek_indexer_compressor_wgate, 536);
    assert_eq!(plan.deepseek_indexer_compressor_norm, 552);
    assert_eq!(plan.deepseek_indexer_weights, 554);
    assert_eq!(plan.rms_mlp, 558);
    assert_eq!(plan.deepseek_indexer_k, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_eq!(plan.deepseek_kv_b_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_eq!(plan.layout_bytes, 584);
}

fn with_tiny_deepseek_v4_descriptor_session(
    layer: CudaHfDecodeChainLayer<'static>,
    max_context_tokens: usize,
    run: impl FnOnce(
        crate::decode::hf_sequence::session::request::CudaHfDecodeSequenceSessionCreateOutput,
    ),
) {
    with_tiny_deepseek_v4_descriptor_session_with_rt(
        layer,
        max_context_tokens,
        CudaHfDecodeSequenceExperimentalRtConfig::default(),
        run,
    );
}

fn with_tiny_deepseek_v4_descriptor_session_with_rt(
    layer: CudaHfDecodeChainLayer<'static>,
    max_context_tokens: usize,
    experimental_rt: CudaHfDecodeSequenceExperimentalRtConfig,
    run: impl FnOnce(
        crate::decode::hf_sequence::session::request::CudaHfDecodeSequenceSessionCreateOutput,
    ),
) {
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V4 descriptor layer");
    assert_ne!(
        plan.deepseek_attention_sink,
        CUDA_HF_SEQUENCE_MISSING_OFFSET
    );
    assert_ne!(plan.deepseek_o_b, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        max_context_tokens,
        rms_eps: 1e-5,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt,
    };

    run(config.create());
}

fn tiny_deepseek_v32_descriptor_layer() -> CudaHfDecodeChainLayer<'static> {
    CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_q_gate: None,
        w_k: &[],
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &[],
        w_o: &[],
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &[],
        w_up: &[],
        w_down: &[],
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: Some(CudaHfDeepSeekLayer {
            mode: CUDA_HF_DEEPSEEK_MODE_V32_MLA_INDEXER,
            flags: CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER,
            hc_mult: 0,
            q_lora_rank: 2,
            kv_lora_rank: 2,
            o_lora_rank: 0,
            o_groups: 0,
            qk_nope_head_dim: 1,
            qk_rope_head_dim: 1,
            v_head_dim: 1,
            compress_ratio: 1,
            index_topk: 0,
            index_n_heads: 2,
            index_head_dim: 2,
            router_num_groups: 1,
            router_topk_groups: 1,
            routed_scaling_factor: 1.0,
        }),
        mlp_kind: 0,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: true,
        attention_kind: CUDA_HF_ATTENTION_DEEPSEEK_MLA,
    }
}

fn tiny_deepseek_v4_descriptor_layer() -> CudaHfDecodeChainLayer<'static> {
    CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_q_gate: None,
        w_k: &[],
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &[],
        w_o: &[],
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &[],
        w_up: &[],
        w_down: &[],
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: Some(CudaHfDeepSeekLayer {
            mode: CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER,
            flags: CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR
                | CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER
                | CUDA_HF_DEEPSEEK_FLAG_MOE
                | CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS,
            hc_mult: 2,
            q_lora_rank: 2,
            kv_lora_rank: 1,
            o_lora_rank: 2,
            o_groups: 2,
            qk_nope_head_dim: 1,
            qk_rope_head_dim: 1,
            v_head_dim: 2,
            compress_ratio: 4,
            index_topk: 4,
            index_n_heads: 1,
            index_head_dim: 2,
            router_num_groups: 0,
            router_topk_groups: 0,
            routed_scaling_factor: 1.0,
        }),
        mlp_kind: CUDA_HF_MLP_SPARSE_MOE,
        moe_intermediate: 4,
        shared_expert_intermediate: 2,
        num_experts: 2,
        experts_per_token: 1,
        norm_topk_prob: true,
        attention_kind: CUDA_HF_ATTENTION_DEEPSEEK_MLA,
    }
}

fn tiny_deepseek_v4_swa_dense_descriptor_layer() -> CudaHfDecodeChainLayer<'static> {
    CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_q_gate: None,
        w_k: &[],
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &[],
        w_o: &[],
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &[],
        w_up: &[],
        w_down: &[],
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: Some(CudaHfDeepSeekLayer {
            mode: CUDA_HF_DEEPSEEK_MODE_V4_SWA,
            flags: 0,
            hc_mult: 2,
            q_lora_rank: 2,
            kv_lora_rank: 0,
            o_lora_rank: 2,
            o_groups: 1,
            qk_nope_head_dim: 1,
            qk_rope_head_dim: 1,
            v_head_dim: 2,
            compress_ratio: 1,
            index_topk: 0,
            index_n_heads: 0,
            index_head_dim: 0,
            router_num_groups: 0,
            router_topk_groups: 0,
            routed_scaling_factor: 1.0,
        }),
        mlp_kind: CUDA_HF_MLP_DENSE,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: false,
        attention_kind: CUDA_HF_ATTENTION_DEEPSEEK_MLA,
    }
}

#[test]
fn query_gate_footprint_counts_optional_projection() {
    let zero = 0x0000;
    let embeddings = vec![zero; 8 * 4];
    let rms = vec![zero; 4];
    let attn = vec![zero; 4 * 4];
    let q_gate = vec![zero; 4 * 4];
    let gate = vec![zero; 8 * 4];
    let down = vec![zero; 4 * 8];
    let lm_head = vec![zero; 8 * 4];
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &attn,
        w_q_gate: Some(&q_gate),
        w_k: &attn,
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &attn,
        w_o: &attn,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &gate,
        w_up: &gate,
        w_down: &down,
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: None,
        mlp_kind: 0,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: false,
        attention_kind: crate::decode::hf_chain::layer::CUDA_HF_ATTENTION_FULL,
    };
    let layers = [layer];
    let request = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 1,
        kv_heads: 1,
        head_dim: 4,
        intermediate: 8,
        vocab_size: 8,
        steps: 4,
        seed_token: 0,
        prompt_tokens: &[0],
        eos_token: None,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &embeddings,
        layers: &layers,
        final_norm_weight: &rms,
        lm_head: &lm_head,
        weight_plan: None,
        weight_blocks: &[],
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    };

    let footprint = estimate_sequence_footprint(&request).unwrap();

    assert_eq!(footprint.resident_weight_bytes, 504);
    assert_eq!(footprint.layout_bytes, 584);
}

#[test]
fn linear_gdn_moe_footprint_counts_state_and_scratch() {
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_q_gate: None,
        w_k: &[],
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &[],
        w_o: &[],
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &[],
        w_up: &[],
        w_down: &[],
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: Some(CudaHfLinearGdnLayer {
            key_heads: 1,
            value_heads: 1,
            key_head_dim: 2,
            value_head_dim: 3,
            conv_kernel: 4,
            w_conv: &[],
            w_qkv: &[],
            w_z: &[],
            w_b: &[],
            w_a: &[],
            dt_bias: &[],
            a_log: &[],
            norm_weight: &[],
            w_out: &[],
        }),
        deepseek: None,
        mlp_kind: CUDA_HF_MLP_SPARSE_MOE,
        moe_intermediate: 3,
        shared_expert_intermediate: 0,
        num_experts: 2,
        experts_per_token: 1,
        norm_topk_prob: true,
        attention_kind: CUDA_HF_ATTENTION_LINEAR_GDN,
    };
    let layers = [layer];
    let request = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 8,
        vocab_size: 4,
        steps: 2,
        seed_token: 0,
        prompt_tokens: &[0],
        eos_token: None,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: 436,
            gpu_resident_weight_bytes: 436,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: 1,
        }),
        weight_blocks: &[],
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    };

    let footprint = estimate_sequence_footprint(&request).unwrap();

    assert_eq!(footprint.resident_weight_bytes, 436);
    assert_eq!(footprint.layout_bytes, 584);
    assert_eq!(footprint.scratch_bytes, 276);
    assert_eq!(footprint.resident_kv_bytes, 128);
    assert_eq!(footprint.device_arena_bytes, 1624);
}

fn assert_raw_descriptor_decode_matches_request(sampler: CudaHfDecodeSamplerConfig) {
    let expected = run_declared_descriptor_decode(sampler);
    if expected.status != SmokeStatus::Ok {
        assert_eq!(expected.status, SmokeStatus::Unavailable);
        return;
    }

    let Some((out, output_tokens)) = run_null_legacy_descriptor_decode(sampler.to_ffi()) else {
        panic!("raw FFI descriptor decode skipped after request decode succeeded");
    };
    assert_eq!(out.status, 0);
    assert_eq!(
        out.descriptor_gpu_resident_h2d_bytes,
        expected.descriptor_gpu_resident_h2d_bytes
    );
    assert_eq!(
        out.descriptor_gpu_staged_h2d_bytes,
        expected.descriptor_gpu_staged_h2d_bytes
    );
    assert_eq!(out.observed_tokens as usize, expected.tokens.len());
    assert_eq!(out.observed_token_hash, expected.observed_token_hash);
    assert_eq!(
        &output_tokens[..expected.tokens.len()],
        expected.tokens.as_slice()
    );
}

fn run_declared_descriptor_decode(
    sampler: CudaHfDecodeSamplerConfig,
) -> CudaHfDecodeSequenceSummary {
    let weights = tiny_descriptor_weights();
    let weight_blocks = weights.blocks();
    let marker_layer = descriptor_marker_layer();
    let layers = [marker_layer];
    CudaHfDecodeSequenceRequest {
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
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
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
        sampler,
    }
    .run()
}

fn descriptor_marker_layer() -> CudaHfDecodeChainLayer<'static> {
    CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_q_gate: None,
        w_k: &[],
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &[],
        w_o: &[],
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &[],
        w_up: &[],
        w_down: &[],
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: None,
        mlp_kind: 0,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: false,
        attention_kind: crate::decode::hf_chain::layer::CUDA_HF_ATTENTION_FULL,
    }
}
