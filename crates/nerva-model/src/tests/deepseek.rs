use crate::hf::architecture::HfArchitectureKind;
use crate::hf::contract::validate_exact_runtime_contract;
use crate::hf::deepseek::{
    deepseek_mla_dimensions, plan_deepseek_vllm_kv_cache,
    plan_deepseek_vllm_kv_cache_with_block_size,
};
use crate::hf::deepseek_runtime::{
    DEEPSEEK_V4_MHC_AUTO_WARMUP_MAX_TOKENS, DeepSeekAttentionExecutionKind,
    deepseek_execution_unit_coverage, deepseek_implemented_primitives,
    deepseek_layer_execution_plan, deepseek_runtime_weight_contract, deepseek_v4_mhc_pre_num_split,
    deepseek_v4_mhc_warmup_token_sizes, plan_deepseek_v4_mhc_warmup,
    validate_deepseek_exact_runtime_contract,
};
use crate::hf::metadata::{HfAttentionLayerKind, HfMlpLayerKind};
use crate::hf::parser::parse_hf_config_metadata;
use crate::weights::layout::entry::WeightBlockRole;
use crate::weights::layout::plan::plan_hf_weight_layout;
use crate::weights::manifest::build_hf_tensor_manifest;
use nerva_core::types::dtype::DType;
use nerva_core::types::error::NervaError;

#[test]
fn parses_deepseek_v3_mla_and_grouped_moe_metadata() {
    let metadata = parse_hf_config_metadata(deepseek_v3_config()).unwrap();

    assert_eq!(metadata.architecture, HfArchitectureKind::DeepSeekV3);
    assert_eq!(metadata.hidden_size, 7168);
    assert_eq!(metadata.num_attention_heads, 128);
    assert_eq!(metadata.num_key_value_heads, 128);
    assert_eq!(metadata.head_dim(), 192);
    assert_eq!(metadata.q_lora_rank, Some(1536));
    assert_eq!(metadata.kv_lora_rank, Some(512));
    assert_eq!(metadata.qk_nope_head_dim, Some(128));
    assert_eq!(metadata.qk_rope_head_dim, Some(64));
    assert_eq!(metadata.v_head_dim, Some(128));
    assert_eq!(metadata.num_experts, Some(256));
    assert_eq!(metadata.num_experts_per_tok, Some(8));
    assert_eq!(metadata.moe_intermediate_size, Some(2048));
    assert_eq!(metadata.shared_expert_intermediate_size, Some(2048));
    assert_eq!(metadata.moe_first_k_dense_replace, Some(3));
    assert_eq!(metadata.moe_layer_freq, Some(1));
    assert_eq!(metadata.num_expert_groups, Some(8));
    assert_eq!(metadata.topk_group, Some(4));
    assert_eq!(metadata.topk_method.as_deref(), Some("noaux_tc"));
    assert_eq!(metadata.scoring_func.as_deref(), Some("sigmoid"));
    assert_eq!(metadata.routed_scaling_factor, Some(2.5));
    let rope_scaling = metadata.rope_scaling.as_ref().unwrap();
    assert_eq!(rope_scaling.rope_type, "deepseek_yarn");
    assert_eq!(rope_scaling.factor, Some(40.0));
    assert_eq!(rope_scaling.original_max_position_embeddings, Some(4096));
    assert_eq!(metadata.num_nextn_predict_layers, Some(1));
    assert_eq!(
        metadata.mlp_layer_types,
        vec![
            HfMlpLayerKind::Dense,
            HfMlpLayerKind::Dense,
            HfMlpLayerKind::Dense,
            HfMlpLayerKind::SparseMoe,
            HfMlpLayerKind::SparseMoe,
            HfMlpLayerKind::SparseMoe,
        ]
    );
    assert!(
        metadata
            .to_json()
            .contains("\"architecture\":\"deepseek_v3\"")
    );
    assert!(metadata.to_json().contains("\"qk_rope_head_dim\":64"));
}

#[test]
fn parses_deepseek_v32_indexer_metadata() {
    let metadata = parse_hf_config_metadata(deepseek_v32_config()).unwrap();

    assert_eq!(metadata.architecture, HfArchitectureKind::DeepSeekV32);
    assert_eq!(metadata.index_topk, Some(2048));
    assert_eq!(metadata.index_topk_freq, None);
    assert_eq!(metadata.index_skip_topk_offset, None);
    assert!(metadata.index_topk_pattern.is_empty());
    assert_eq!(metadata.index_n_heads, Some(64));
    assert_eq!(metadata.index_head_dim, Some(128));
    assert_eq!(metadata.sliding_window, None);
    assert_eq!(metadata.mlp_layer_types[0], HfMlpLayerKind::Dense);
    assert_eq!(metadata.mlp_layer_types[3], HfMlpLayerKind::SparseMoe);
}

#[test]
fn deepseek_v32_layer_plan_matches_vllm_index_topk_frequency() {
    let config = deepseek_v32_config().replace(
        "\"index_topk\": 2048,",
        "\"index_topk\": 2048,\n        \"index_topk_freq\": 2,\n        \"index_skip_topk_offset\": 2,",
    );
    let metadata = parse_hf_config_metadata(&config).unwrap();
    let plan = deepseek_layer_execution_plan(&metadata).unwrap();

    assert_eq!(metadata.index_topk_freq, Some(2));
    assert_eq!(metadata.index_skip_topk_offset, Some(2));
    assert_eq!(
        plan.layers
            .iter()
            .map(|layer| layer.uses_sparse_indexer)
            .collect::<Vec<_>>(),
        vec![true, true, false, true]
    );
    assert_eq!(
        plan.layers
            .iter()
            .map(|layer| layer.indexer_kv_cache_group.as_deref())
            .collect::<Vec<_>>(),
        vec![
            Some("v3_2_sparse_indexer"),
            Some("v3_2_sparse_indexer"),
            None,
            Some("v3_2_sparse_indexer"),
        ]
    );
}

#[test]
fn deepseek_v32_layer_plan_matches_vllm_index_topk_pattern() {
    let config = deepseek_v32_config().replace(
        "\"index_topk\": 2048,",
        "\"index_topk\": 2048,\n        \"index_topk_pattern\": [\"S\", \"D\", \"S\", \"D\"],",
    );
    let metadata = parse_hf_config_metadata(&config).unwrap();
    let plan = deepseek_layer_execution_plan(&metadata).unwrap();

    assert_eq!(
        metadata.index_topk_pattern,
        vec![
            "S".to_string(),
            "D".to_string(),
            "S".to_string(),
            "D".to_string(),
        ]
    );
    assert_eq!(
        plan.layers
            .iter()
            .map(|layer| layer.uses_sparse_indexer)
            .collect::<Vec<_>>(),
        vec![false, true, false, true]
    );
}

#[test]
fn parses_deepseek_v4_flash_metadata() {
    let metadata = parse_hf_config_metadata(deepseek_v4_flash_config()).unwrap();

    assert_eq!(metadata.architecture, HfArchitectureKind::DeepSeekV4);
    assert_eq!(metadata.hidden_size, 4096);
    assert_eq!(metadata.num_attention_heads, 64);
    assert_eq!(metadata.num_key_value_heads, 1);
    assert_eq!(metadata.head_dim(), 512);
    assert_eq!(metadata.q_lora_rank, Some(1024));
    assert_eq!(metadata.o_lora_rank, Some(1024));
    assert_eq!(metadata.o_groups, Some(8));
    assert_eq!(metadata.qk_nope_head_dim, Some(448));
    assert_eq!(metadata.qk_rope_head_dim, Some(64));
    assert_eq!(metadata.v_head_dim, Some(512));
    assert_eq!(metadata.num_experts, Some(256));
    assert_eq!(metadata.num_experts_per_tok, Some(6));
    assert_eq!(metadata.moe_intermediate_size, Some(2048));
    assert_eq!(metadata.shared_expert_intermediate_size, Some(2048));
    assert_eq!(metadata.scoring_func.as_deref(), Some("sqrtsoftplus"));
    assert_eq!(metadata.index_topk, Some(512));
    assert_eq!(metadata.sliding_window, Some(4096));
    assert_eq!(metadata.compress_ratios, vec![0, 0, 4, 128]);
    assert_eq!(metadata.hc_mult, Some(4));
    assert_eq!(metadata.hc_sinkhorn_iters, Some(20));
    assert_eq!(metadata.hc_eps, Some(0.000001));
    assert_eq!(metadata.num_hash_layers, Some(3));
    assert_eq!(metadata.compress_rope_theta, Some(1_000_000.0));
    let rope_scaling = metadata.rope_scaling.as_ref().unwrap();
    assert_eq!(rope_scaling.rope_type, "deepseek_yarn");
    assert_eq!(metadata.swiglu_limit, Some(10.0));
    assert_eq!(metadata.expert_dtype.as_deref(), Some("fp4"));
    assert!(
        metadata
            .mlp_layer_types
            .iter()
            .all(|kind| *kind == HfMlpLayerKind::SparseMoe)
    );
}

#[test]
fn parses_deepseek_v4_vllm_attention_layer_type_aliases() {
    let config = deepseek_v4_flash_config().replace(
        "\"compress_ratios\": [0, 0, 4, 128],",
        "\"layer_types\": [\"sliding_attention\", \"sliding_attention\", \"compressed_sparse_attention\", \"heavily_compressed_attention\"],\n        \"compress_ratios\": [0, 0, 4, 128],",
    );
    let metadata = parse_hf_config_metadata(&config).unwrap();

    assert_eq!(metadata.architecture, HfArchitectureKind::DeepSeekV4);
    assert_eq!(
        metadata.attention_layer_types,
        vec![
            HfAttentionLayerKind::Full,
            HfAttentionLayerKind::Full,
            HfAttentionLayerKind::Full,
            HfAttentionLayerKind::Full,
        ]
    );
    assert_eq!(metadata.compress_ratios, vec![0, 0, 4, 128]);
}

#[test]
fn deepseek_v4_mhc_warmup_plan_matches_vllm_token_sizes_and_split_k() {
    let metadata = parse_hf_config_metadata(deepseek_v4_flash_config()).unwrap();

    assert_eq!(DEEPSEEK_V4_MHC_AUTO_WARMUP_MAX_TOKENS, 16_384);
    assert_eq!(
        deepseek_v4_mhc_warmup_token_sizes(9000, &[3, 64, 8192, 12_000]),
        vec![
            1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 9000
        ]
    );
    assert_eq!(
        deepseek_v4_mhc_warmup_token_sizes(20_000, &[17_000, 2048]),
        vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16_384
        ]
    );
    assert!(deepseek_v4_mhc_warmup_token_sizes(0, &[1]).is_empty());

    assert_eq!(
        deepseek_v4_mhc_pre_num_split(64, 7168, 4, 120).unwrap(),
        112
    );
    assert_eq!(
        deepseek_v4_mhc_pre_num_split(8192, 4096, 4, 120).unwrap(),
        1
    );
    assert!(deepseek_v4_mhc_pre_num_split(0, 4096, 4, 120).is_err());

    let plan = plan_deepseek_v4_mhc_warmup(&metadata, 9000, &[3, 64, 8192, 12_000], 120)
        .expect("DeepSeek V4 mHC warmup should plan for V4 metadata");
    assert_eq!(plan.max_tokens, 9000);
    assert_eq!(plan.hidden_size, 4096);
    assert_eq!(plan.hc_mult, 4);
    assert_eq!(plan.num_sms, 120);
    assert_eq!(plan.token_sizes.first().unwrap().tokens, 1);
    assert_eq!(plan.token_sizes.last().unwrap().tokens, 9000);
    assert_eq!(plan.token_sizes[0].mhc_pre_num_split, 64);
    assert_eq!(plan.token_sizes[7].tokens, 64);
    assert_eq!(plan.token_sizes[7].mhc_pre_num_split, 64);
    assert_eq!(plan.token_sizes[14].tokens, 8192);
    assert_eq!(plan.token_sizes[14].mhc_pre_num_split, 1);

    let v3 = parse_hf_config_metadata(deepseek_v3_config()).unwrap();
    assert!(plan_deepseek_v4_mhc_warmup(&v3, 1024, &[], 120).is_err());
}

#[test]
fn deepseek_v4_coverage_reports_cuda_mhc_sequence_runtime_complete() {
    let metadata = parse_hf_config_metadata(deepseek_v4_flash_config()).unwrap();

    let primitives = deepseek_implemented_primitives(&metadata);
    for primitive in [
        "cuda_deepseek_mhc_pre_api",
        "cuda_deepseek_mhc_pre_smoke",
        "cuda_deepseek_mhc_post_api",
        "cuda_deepseek_mhc_post_smoke",
        "cuda_deepseek_mhc_fused_post_pre_api",
        "cuda_deepseek_mhc_fused_post_pre_smoke",
        "cuda_deepseek_mhc_head_api",
        "cuda_deepseek_mhc_head_smoke",
        "cuda_hf_sequence_deepseek_v4_mhc_sequence_runtime",
        "cuda_hf_sequence_deepseek_v4_mhc_head_final_norm_runtime",
        "cuda_hf_sequence_deepseek_v4_mhc_native_profile_runtime",
        "cuda_hf_sequence_deepseek_packed_kv_footprint_accounting",
    ] {
        assert!(
            primitives.iter().any(|item| item == primitive),
            "missing DeepSeek V4 mHC primitive coverage entry: {primitive}"
        );
    }

    let coverage = deepseek_execution_unit_coverage(&metadata);
    let mhc = coverage
        .iter()
        .find(|unit| unit.unit == "deepseek_v4_mhc_pre_post_head")
        .expect("DeepSeek V4 should report mHC coverage");
    assert_eq!(mhc.status, "complete");
    assert!(
        mhc.validated_primitives
            .iter()
            .any(|item| item == "cuda_deepseek_mhc_fused_post_pre_api")
    );
    assert!(
        mhc.validated_primitives
            .iter()
            .any(|item| item == "cuda_hf_sequence_deepseek_v4_mhc_sequence_runtime")
    );
    assert!(mhc.remaining_gaps.is_empty());

    let swa = coverage
        .iter()
        .find(|unit| unit.unit == "deepseek_v4_mla_swa_cache")
        .expect("DeepSeek V4 should report SWA cache coverage");
    assert_eq!(swa.status, "partial");
    for primitive in [
        "cuda_hf_sequence_deepseek_v4_swa_parallel_head_attention_runtime",
        "cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_nonzero_page_contents",
        "cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_fullsize_page_contents",
        "cuda_hf_sequence_deepseek_packed_kv_footprint_accounting",
    ] {
        assert!(
            primitives.iter().any(|item| item == primitive),
            "missing DeepSeek V4 SWA primitive coverage entry: {primitive}"
        );
        assert!(
            swa.validated_primitives
                .iter()
                .any(|item| item == primitive),
            "missing DeepSeek V4 SWA validated primitive: {primitive}"
        );
    }

    let compressed = coverage
        .iter()
        .find(|unit| unit.unit == "deepseek_v4_fp8_ds_mla_cache")
        .expect("DeepSeek V4 should report compressed MLA cache coverage");
    assert_eq!(compressed.status, "partial");
    assert!(
        compressed
            .validated_primitives
            .iter()
            .any(|item| item == "cuda_hf_sequence_deepseek_v4_c128_parallel_head_attention_runtime"),
        "DeepSeek V4 C128 parallel attention primitive must be reported"
    );
    let compressor = coverage
        .iter()
        .find(|unit| unit.unit == "deepseek_v4_c4_c128_compressor")
        .expect("DeepSeek V4 should report C4/C128 compressor coverage");
    assert!(
        compressor
            .validated_primitives
            .iter()
            .any(|item| item == "cuda_hf_sequence_deepseek_v4_parallel_compressed_kv_pack_runtime"),
        "DeepSeek V4 parallel compressed KV pack primitive must be reported"
    );
    assert!(
        compressor
            .validated_primitives
            .iter()
            .any(|item| item == "cuda_hf_sequence_deepseek_v4_parallel_indexer_kv_pack_runtime"),
        "DeepSeek V4 parallel indexer KV pack primitive must be reported"
    );
    assert!(
        compressor
            .validated_primitives
            .iter()
            .any(|item| item == "cuda_hf_sequence_deepseek_v4_c4_indexer_kv_page_contents"),
        "DeepSeek V4 C4 indexer KV page contents primitive must be reported"
    );
    let sparse_indexer = coverage
        .iter()
        .find(|unit| unit.unit == "deepseek_v4_sparse_indexer")
        .expect("DeepSeek V4 should report sparse indexer coverage");
    assert!(
        sparse_indexer
            .validated_primitives
            .iter()
            .any(|item| item
                == "cuda_hf_sequence_deepseek_v4_c4_sparse_parallel_head_attention_runtime"),
        "DeepSeek V4 C4 sparse parallel attention primitive must be reported"
    );

    let parallel_attention = coverage
        .iter()
        .find(|unit| unit.unit == "deepseek_v4_parallel_attention_gemm_streams")
        .expect("DeepSeek V4 should report parallel attention/GEMM stream coverage");
    for primitive in [
        "cuda_hf_sequence_deepseek_v4_aux_qk_projection_runtime",
        "cuda_hf_sequence_deepseek_v4_aux_compressor_indexer_runtime",
        "cuda_hf_sequence_deepseek_v4_grouped_output_projection_runtime",
    ] {
        assert!(
            parallel_attention
                .validated_primitives
                .iter()
                .any(|item| item == primitive),
            "missing DeepSeek V4 aux stream primitive: {primitive}"
        );
    }

    let megamoe = coverage
        .iter()
        .find(|unit| unit.unit == "deepseek_v4_megamoe_int8_fp4_experts")
        .expect("DeepSeek V4 should report MegaMoE coverage");
    assert_eq!(megamoe.status, "partial");
    for primitive in [
        "cuda_deepseek_megamoe_prepare_api",
        "cuda_deepseek_megamoe_prepare_smoke",
        "cuda_deepseek_megamoe_eplb_mapping_api",
        "cuda_deepseek_megamoe_eplb_mapping_smoke",
        "cuda_deepseek_megamoe_fp8_fp4_expert_api",
        "cuda_deepseek_megamoe_fp8_fp4_expert_smoke",
        "deepseek_full_routed_moe_reference",
        "cuda_hf_sequence_deepseek_v4_mxfp4_expert_gate_up_runtime",
        "cuda_hf_sequence_deepseek_v4_mxfp4_expert_down_runtime",
        "cuda_hf_sequence_deepseek_v4_parallel_sparse_moe_runtime",
    ] {
        assert!(
            primitives.iter().any(|item| item == primitive),
            "missing DeepSeek V4 MegaMoE primitive coverage entry: {primitive}"
        );
        assert!(
            megamoe
                .validated_primitives
                .iter()
                .any(|item| item == primitive),
            "missing DeepSeek V4 MegaMoE validated primitive: {primitive}"
        );
    }

    let router = coverage
        .iter()
        .find(|unit| unit.unit == "deepseek_v4_hash_and_bias_router")
        .expect("DeepSeek V4 should report hash/bias router coverage");
    assert!(
        router
            .validated_primitives
            .iter()
            .any(|item| item == "deepseek_v4_full_routed_moe_hash_reference")
    );
    assert!(
        router
            .validated_primitives
            .iter()
            .any(|item| item == "cuda_hf_sequence_deepseek_v4_sparse_moe_route_runtime")
    );

    let parity = coverage
        .iter()
        .find(|unit| unit.unit == "deepseek_v4_vllm_e2e_parity")
        .expect("DeepSeek V4 should keep an explicit vLLM parity gate");
    assert_eq!(parity.status, "partial");
    assert!(
        parity
            .remaining_gaps
            .iter()
            .any(|gap| gap.contains("/root/vllm")),
        "vLLM parity gate must point at the local vLLM checkout"
    );
}

#[test]
fn deepseek_v32_projection_coverage_tracks_live_scale_runtime() {
    let metadata = parse_hf_config_metadata(deepseek_v32_config()).unwrap();
    let primitives = deepseek_implemented_primitives(&metadata);
    let coverage = deepseek_execution_unit_coverage(&metadata);
    let unit = coverage
        .iter()
        .find(|unit| unit.unit == "deepseek_v3_block_fp8_projection_gemm")
        .expect("DeepSeek V3.2 should report projection GEMM coverage");

    assert_eq!(unit.status, "partial");
    for primitive in [
        "cuda_hf_sequence_deepseek_v32_sparse_mla_kv_b_scale_runtime",
        "cuda_hf_sequence_deepseek_v32_output_projection_scale_logits_runtime",
        "cuda_hf_sequence_deepseek_v32_q_a_kv_a_q_b_scale_sparse_decode_runtime",
        "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_token4_weight_reuse",
        "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_row8_token4_input_reuse",
    ] {
        assert!(
            primitives.iter().any(|item| item == primitive),
            "V3.2 implemented primitive list must include {primitive}"
        );
        assert!(
            unit.validated_primitives
                .iter()
                .any(|item| item == primitive),
            "V3.2 projection coverage must include {primitive}"
        );
    }
    assert!(
        unit.remaining_gaps
            .iter()
            .all(|gap| !gap.contains("consume packed DeepSeek q_a/kv_a/q_b/kv_b/o")),
        "coverage should not claim all packed projection scales are ignored after KV-B is tested"
    );
    assert!(
        unit.remaining_gaps
            .iter()
            .all(|gap| !gap.contains("q_a/kv_a/q_b projection scale offsets")),
        "q_a/kv_a/q_b scale offsets now have sparse decode output coverage"
    );
    assert!(
        unit.remaining_gaps
            .iter()
            .any(|gap| gap.contains("vLLM-class tensor-core/DeepGEMM")),
        "remaining projection work should name the actual kernel-class gap"
    );
    assert!(
        unit.remaining_gaps
            .iter()
            .all(|gap| !gap.contains("CTA-per-row")),
        "projection gap should not describe the row-tiled kernel as CTA-per-row"
    );
    assert!(
        unit.remaining_gaps
            .iter()
            .all(|gap| !gap.contains("fuse block-FP8 dequant with projection GEMM")),
        "projection coverage should not say fusion is missing after fused tile coverage exists"
    );
}

#[test]
fn deepseek_v32_sparse_indexer_coverage_tracks_vllm_topk_schedule() {
    let metadata = parse_hf_config_metadata(deepseek_v32_config()).unwrap();
    let primitives = deepseek_implemented_primitives(&metadata);
    let coverage = deepseek_execution_unit_coverage(&metadata);
    let unit = coverage
        .iter()
        .find(|unit| unit.unit == "deepseek_v32_sparse_attention_indexer")
        .expect("DeepSeek V3.2 should report sparse indexer coverage");

    assert!(
        primitives
            .iter()
            .any(|item| item == "deepseek_v32_vllm_index_topk_skip_schedule")
    );
    assert!(
        unit.validated_primitives
            .iter()
            .any(|item| item == "deepseek_v32_vllm_index_topk_skip_schedule")
    );
}

#[test]
fn deepseek_vllm_kv_plan_matches_v3_and_v32_mla_cache_contracts() {
    let v3 = parse_hf_config_metadata(deepseek_v3_config()).unwrap();
    let dims = deepseek_mla_dimensions(&v3).unwrap();
    assert_eq!(dims.kv_lora_rank, 512);
    assert_eq!(dims.qk_rope_head_dim, 64);
    assert_eq!(dims.semantic_head_size, 576);

    let v3_plan = plan_deepseek_vllm_kv_cache(&v3, "bfloat16").unwrap();
    assert_eq!(v3_plan.default_block_size, 64);
    assert_eq!(v3_plan.groups.len(), 1);
    assert_eq!(v3_plan.groups[0].name, "v3_main_mla");
    assert_eq!(v3_plan.groups[0].layers, 6);
    assert_eq!(v3_plan.groups[0].spec.head_size, 576);
    assert_eq!(v3_plan.groups[0].spec.kv_quant_mode, "none");
    assert_eq!(v3_plan.groups[0].spec.page_size_padded, None);
    assert!(!v3_plan.groups[0].spec.indexes_kv_by_block_stride);
    assert_eq!(v3_plan.groups[0].spec.real_page_size_bytes, 64 * 576 * 2);

    let v32 = parse_hf_config_metadata(deepseek_v32_config()).unwrap();
    let v32_plan = plan_deepseek_vllm_kv_cache(&v32, "fp8_ds_mla").unwrap();
    assert_eq!(v32_plan.groups.len(), 2);
    assert_eq!(v32_plan.groups[0].name, "v3_2_main_mla");
    assert_eq!(v32_plan.groups[0].spec.dtype, DType::U8);
    assert_eq!(v32_plan.groups[0].spec.kv_quant_mode, "none");
    assert_eq!(v32_plan.groups[0].spec.page_size_padded, None);
    assert_eq!(v32_plan.groups[0].spec.real_page_size_bytes, 64 * 656);
    assert_eq!(v32_plan.groups[1].name, "v3_2_sparse_indexer");
    assert_eq!(v32_plan.groups[1].spec.head_size, 132);
    assert_eq!(v32_plan.groups[1].spec.page_size_padded, None);
    assert!(!v32_plan.groups[1].spec.indexes_kv_by_block_stride);
    assert_eq!(v32_plan.groups[1].spec.real_page_size_bytes, 64 * 132);
    assert!(
        v32_plan
            .to_json()
            .contains("/root/vllm/vllm/v1/attention/backends/mla/indexer.py")
    );
}

#[test]
fn deepseek_vllm_kv_plan_matches_v4_sparse_swa_and_indexer_contracts() {
    let metadata = parse_hf_config_metadata(deepseek_v4_flash_config()).unwrap();
    let dims = deepseek_mla_dimensions(&metadata).unwrap();
    assert_eq!(dims.kv_lora_rank, 512);
    assert_eq!(dims.qk_nope_head_dim, 448);
    assert_eq!(dims.qk_rope_head_dim, 64);
    assert_eq!(dims.v_head_dim, 512);
    assert_eq!(dims.semantic_head_size, 512);

    let plan = plan_deepseek_vllm_kv_cache(&metadata, "fp8_ds_mla").unwrap();
    assert_eq!(plan.default_block_size, 256);
    assert_eq!(plan.groups.len(), 7);

    let swa = &plan.groups[0];
    assert_eq!(swa.name, "v4_swa");
    assert_eq!(swa.layers, 4);
    assert_eq!(swa.spec.kind, "sliding_window_mla");
    assert_eq!(swa.spec.block_size, 64);
    assert_eq!(swa.spec.sliding_window, Some(4096));
    assert_eq!(swa.spec.real_page_size_bytes, 64 * 584);
    assert_eq!(swa.spec.page_size_padded, Some(37440));
    assert_eq!(swa.spec.page_size_bytes, 37440);
    assert!(!swa.spec.indexes_kv_by_block_stride);

    let c4 = &plan.groups[1];
    assert_eq!(c4.name, "v4_c4_mla");
    assert_eq!(c4.layers, 1);
    assert_eq!(c4.spec.storage_block_size, 64);
    assert_eq!(c4.spec.real_page_size_bytes, 64 * 584);
    assert_eq!(c4.spec.page_size_padded, Some(37440));
    assert_eq!(c4.spec.alignment, Some(576));
    assert_eq!(c4.spec.kv_quant_mode, "none");

    let c4_indexer = &plan.groups[2];
    assert_eq!(c4_indexer.name, "v4_c4_mla_indexer");
    assert_eq!(c4_indexer.spec.head_size, 132);
    assert_eq!(c4_indexer.spec.storage_block_size, 64);
    assert_eq!(c4_indexer.spec.real_page_size_bytes, 64 * 132);
    assert_eq!(c4_indexer.spec.page_size_padded, Some(8640));
    assert_eq!(c4_indexer.spec.page_size_bytes, 8640);

    let c4_compressor = &plan.groups[3];
    assert_eq!(c4_compressor.name, "v4_c4_compressor_state");
    assert_eq!(c4_compressor.layers, 1);
    assert_eq!(c4_compressor.spec.kind, "sliding_window_mla");
    assert_eq!(c4_compressor.spec.block_size, 4);
    assert_eq!(c4_compressor.spec.head_size, 2048);
    assert_eq!(c4_compressor.spec.sliding_window, Some(8));
    assert_eq!(c4_compressor.spec.real_page_size_bytes, 4 * 2048 * 4);
    assert_eq!(c4_compressor.spec.page_size_padded, Some(32832));
    assert_eq!(c4_compressor.spec.page_size_bytes, 32832);

    let c4_indexer_compressor = &plan.groups[4];
    assert_eq!(c4_indexer_compressor.name, "v4_c4_indexer_compressor_state");
    assert_eq!(c4_indexer_compressor.layers, 1);
    assert_eq!(c4_indexer_compressor.spec.block_size, 4);
    assert_eq!(c4_indexer_compressor.spec.head_size, 512);
    assert_eq!(c4_indexer_compressor.spec.sliding_window, Some(8));
    assert_eq!(c4_indexer_compressor.spec.real_page_size_bytes, 4 * 512 * 4);
    assert_eq!(c4_indexer_compressor.spec.page_size_padded, Some(8640));
    assert_eq!(c4_indexer_compressor.spec.page_size_bytes, 8640);

    let c128 = &plan.groups[5];
    assert_eq!(c128.name, "v4_c128_mla");
    assert_eq!(c128.layers, 1);
    assert_eq!(c128.spec.storage_block_size, 2);
    assert_eq!(c128.spec.real_page_size_bytes, 2 * 584);
    assert_eq!(c128.spec.page_size_padded, Some(1728));
    assert_eq!(c128.spec.page_size_bytes, 1728);

    let c128_compressor = &plan.groups[6];
    assert_eq!(c128_compressor.name, "v4_c128_compressor_state");
    assert_eq!(c128_compressor.layers, 1);
    assert_eq!(c128_compressor.spec.block_size, 8);
    assert_eq!(c128_compressor.spec.head_size, 1024);
    assert_eq!(c128_compressor.spec.sliding_window, Some(128));
    assert_eq!(c128_compressor.spec.real_page_size_bytes, 8 * 1024 * 4);
    assert_eq!(c128_compressor.spec.page_size_padded, Some(32832));
    assert_eq!(c128_compressor.spec.page_size_bytes, 32832);
    assert!(
        plan.to_json()
            .contains("/root/vllm/vllm/models/deepseek_v4/sparse_mla.py")
    );
    assert!(
        plan.to_json()
            .contains("\"indexes_kv_by_block_stride\":false")
    );

    let packed = plan
        .packed_layout
        .as_ref()
        .expect("DeepSeek V4 should expose vLLM packed KV layout");
    assert_eq!(packed.total_bytes_per_block, 37_440 + 8_640 + 1_728);
    assert_eq!(packed.tensors.len(), 3);
    assert_eq!(packed.tensors[0].page_size_bytes, 37_440);
    assert_eq!(packed.tensors[0].slot_index, 0);
    assert_eq!(packed.tensors[0].offset_bytes, 0);
    assert_eq!(packed.tensors[0].block_stride_bytes, 47_808);
    assert_eq!(packed.tensors[0].shared_by.len(), 7);
    assert!(
        packed.tensors[0]
            .shared_by
            .iter()
            .any(|name| name == "model.layers.2.self_attn.compressor.state_cache")
    );
    assert!(
        packed.tensors[0]
            .shared_by
            .iter()
            .any(|name| name == "model.layers.3.self_attn.compressor.state_cache")
    );
    assert_eq!(packed.tensors[1].page_size_bytes, 8_640);
    assert_eq!(packed.tensors[1].offset_bytes, 37_440);
    assert_eq!(packed.tensors[1].block_stride_bytes, 47_808);
    assert_eq!(packed.tensors[1].shared_by.len(), 2);
    assert!(
        packed.tensors[1]
            .shared_by
            .iter()
            .any(|name| name == "model.layers.2.self_attn.indexer.compressor.state_cache")
    );
    assert_eq!(packed.tensors[2].page_size_bytes, 1_728);
    assert_eq!(packed.tensors[2].offset_bytes, 46_080);
    assert_eq!(packed.tensors[2].block_stride_bytes, 47_808);
    assert_eq!(
        packed.tensors[2].shared_by,
        vec!["model.layers.3.self_attn".to_string()]
    );
    assert!(plan.to_json().contains("\"packed_layout\""));
}

#[test]
fn deepseek_layer_execution_plan_matches_vllm_v3_v32_and_v4_modes() {
    let v3 = parse_hf_config_metadata(deepseek_v3_config()).unwrap();
    let v3_plan = deepseek_layer_execution_plan(&v3).unwrap();
    assert_eq!(v3_plan.cache_dtype_str, "bfloat16");
    assert_eq!(v3_plan.default_block_size, 64);
    assert_eq!(v3_plan.layers.len(), 6);
    assert!(
        v3_plan
            .layers
            .iter()
            .all(|layer| layer.attention_kind == DeepSeekAttentionExecutionKind::V3Mla)
    );
    assert!(
        v3_plan
            .layers
            .iter()
            .all(|layer| layer.primary_kv_cache_group == "v3_main_mla")
    );
    assert!(!v3_plan.layers[0].uses_moe);
    assert!(v3_plan.layers[3].uses_moe);

    let v32 = parse_hf_config_metadata(deepseek_v32_config()).unwrap();
    let v32_plan = deepseek_layer_execution_plan(&v32).unwrap();
    assert_eq!(v32_plan.cache_dtype_str, "fp8_ds_mla");
    assert_eq!(
        v32_plan.layers[0].attention_kind.as_str(),
        "deepseek_v3_2_mla_with_indexer"
    );
    assert_eq!(v32_plan.layers[0].primary_kv_cache_group, "v3_2_main_mla");
    assert_eq!(
        v32_plan.layers[0].indexer_kv_cache_group.as_deref(),
        Some("v3_2_sparse_indexer")
    );
    assert!(
        v32_plan
            .layers
            .iter()
            .all(|layer| layer.uses_sparse_indexer)
    );
    assert!(
        v32_plan
            .layers
            .iter()
            .all(|layer| layer.uses_compressed_indexer_cache)
    );

    let v4 = parse_hf_config_metadata(deepseek_v4_flash_config()).unwrap();
    let v4_plan = deepseek_layer_execution_plan(&v4).unwrap();
    assert_eq!(v4_plan.cache_dtype_str, "fp8_ds_mla");
    assert_eq!(v4_plan.default_block_size, 256);
    assert_eq!(
        v4_plan
            .layers
            .iter()
            .map(|layer| layer.compress_ratio)
            .collect::<Vec<_>>(),
        vec![1, 1, 4, 128]
    );
    assert_eq!(
        v4_plan
            .layers
            .iter()
            .map(|layer| layer.index_topk)
            .collect::<Vec<_>>(),
        vec![512, 512, 512, 512]
    );
    assert_eq!(
        v4_plan
            .layers
            .iter()
            .map(|layer| layer.attention_kind)
            .collect::<Vec<_>>(),
        vec![
            DeepSeekAttentionExecutionKind::V4SlidingWindowMla,
            DeepSeekAttentionExecutionKind::V4SlidingWindowMla,
            DeepSeekAttentionExecutionKind::V4CompressedMlaWithSparseIndexer,
            DeepSeekAttentionExecutionKind::V4CompressedMla,
        ]
    );
    assert_eq!(v4_plan.layers[0].primary_kv_cache_group, "v4_swa");
    assert!(v4_plan.layers[0].uses_sliding_window_cache);
    assert_eq!(v4_plan.layers[2].primary_kv_cache_group, "v4_c4_mla");
    assert_eq!(
        v4_plan.layers[2].indexer_kv_cache_group.as_deref(),
        Some("v4_c4_mla_indexer")
    );
    assert_eq!(
        v4_plan.layers[2].compressor_state_kv_cache_group.as_deref(),
        Some("v4_c4_compressor_state")
    );
    assert_eq!(
        v4_plan.layers[2]
            .indexer_compressor_state_kv_cache_group
            .as_deref(),
        Some("v4_c4_indexer_compressor_state")
    );
    assert!(v4_plan.layers[2].uses_sparse_indexer);
    assert!(v4_plan.layers[2].uses_compressor);
    assert_eq!(v4_plan.layers[3].primary_kv_cache_group, "v4_c128_mla");
    assert_eq!(v4_plan.layers[3].indexer_kv_cache_group.as_deref(), None);
    assert_eq!(
        v4_plan.layers[3].compressor_state_kv_cache_group.as_deref(),
        Some("v4_c128_compressor_state")
    );
    assert!(!v4_plan.layers[3].uses_sparse_indexer);
    assert!(!v4_plan.layers[3].uses_compressed_indexer_cache);
    assert!(v4_plan.layers[3].uses_compressor);
    assert!(v4_plan.layers[0].uses_hash_router);
    assert!(v4_plan.layers[2].uses_hash_router);
    assert!(!v4_plan.layers[3].uses_hash_router);
}

#[test]
fn deepseek_runtime_weight_contract_binds_execution_modes_to_layer_roles() {
    let v3 = parse_hf_config_metadata(deepseek_v3_config()).unwrap();
    let v3_contract = deepseek_runtime_weight_contract(&v3).unwrap();
    assert!(has_role(
        &v3_contract.layers[0],
        WeightBlockRole::DeepSeekQALoraProjection
    ));
    assert!(has_role(
        &v3_contract.layers[0],
        WeightBlockRole::GateScaleInv
    ));
    assert!(!has_role(
        &v3_contract.layers[0],
        WeightBlockRole::RouterProjection
    ));
    assert!(has_role(
        &v3_contract.layers[3],
        WeightBlockRole::RouterCorrectionBias
    ));
    assert!(has_role(
        &v3_contract.layers[3],
        WeightBlockRole::ExpertGateScaleInv
    ));

    let v32 = parse_hf_config_metadata(deepseek_v32_config()).unwrap();
    let v32_contract = deepseek_runtime_weight_contract(&v32).unwrap();
    assert!(
        v32_contract
            .layers
            .iter()
            .all(|layer| has_role(layer, WeightBlockRole::DeepSeekIndexerWeightsProjection))
    );
    assert!(
        v32_contract
            .layers
            .iter()
            .all(|layer| has_role(layer, WeightBlockRole::DeepSeekIndexerKeyNormBias))
    );

    let v4 = parse_hf_config_metadata(deepseek_v4_flash_config()).unwrap();
    let v4_contract = deepseek_runtime_weight_contract(&v4).unwrap();
    assert!(has_role(
        &v4_contract.layers[0],
        WeightBlockRole::DeepSeekV4HcAttnBase
    ));
    assert!(has_role(
        &v4_contract.layers[0],
        WeightBlockRole::DeepSeekV4HashRouteTable
    ));
    assert!(!has_role(
        &v4_contract.layers[0],
        WeightBlockRole::DeepSeekV4CompressorWkvProjection
    ));
    assert!(has_role(
        &v4_contract.layers[2],
        WeightBlockRole::DeepSeekV4CompressorWkvProjection
    ));
    assert!(has_role(
        &v4_contract.layers[2],
        WeightBlockRole::DeepSeekV4IndexerWqBProjection
    ));
    assert!(has_role(
        &v4_contract.layers[2],
        WeightBlockRole::DeepSeekV4IndexerCompressorWkvProjection
    ));
    assert!(has_role(
        &v4_contract.layers[3],
        WeightBlockRole::DeepSeekV4CompressorWkvProjection
    ));
    assert!(!has_role(
        &v4_contract.layers[3],
        WeightBlockRole::DeepSeekV4IndexerWqBProjection
    ));
    assert!(has_role(
        &v4_contract.layers[3],
        WeightBlockRole::RouterCorrectionBias
    ));
    assert!(has_role(
        &v4_contract.layers[3],
        WeightBlockRole::DeepSeekV4ExpertGateScale
    ));
}

#[test]
fn deepseek_vllm_kv_plan_rejects_invalid_block_and_cache_dtype_combinations() {
    let v3 = parse_hf_config_metadata(deepseek_v3_config()).unwrap();
    let error = plan_deepseek_vllm_kv_cache(&v3, "fp8_ds_mla").unwrap_err();
    assert!(format!("{error:?}").contains("DeepSeek V3.2"));

    let v4 = parse_hf_config_metadata(deepseek_v4_flash_config()).unwrap();
    let error = plan_deepseek_vllm_kv_cache_with_block_size(&v4, "fp8_ds_mla", 64).unwrap_err();
    assert!(format!("{error:?}").contains("compress_ratio 128"));
}

#[test]
fn recognized_deepseek_configs_fail_exact_runtime_contract_precisely() {
    for config in [
        deepseek_v3_config(),
        deepseek_v32_config(),
        deepseek_v4_flash_config(),
    ] {
        let metadata = parse_hf_config_metadata(config).unwrap();
        let NervaError::InvalidArgument { reason } =
            validate_exact_runtime_contract(&metadata).unwrap_err()
        else {
            panic!("DeepSeek runtime contract should fail with InvalidArgument");
        };

        assert!(reason.contains("DeepSeek MLA attention"), "{reason}");
        assert!(reason.contains("block-quantized"), "{reason}");
        assert!(reason.contains("/root/vllm"), "{reason}");
        assert_eq!(
            format!(
                "{:?}",
                validate_deepseek_exact_runtime_contract(&metadata).unwrap_err()
            ),
            format!("{:?}", NervaError::InvalidArgument { reason }),
            "general exact runtime contract should delegate to the DeepSeek coverage-driven gate"
        );
    }

    assert!(
        plan_hf_weight_layout(&parse_hf_config_metadata(deepseek_v3_config()).unwrap()).is_ok()
    );
    assert!(
        plan_hf_weight_layout(&parse_hf_config_metadata(deepseek_v32_config()).unwrap()).is_ok()
    );
    assert!(
        plan_hf_weight_layout(&parse_hf_config_metadata(deepseek_v4_flash_config()).unwrap())
            .is_ok()
    );
}

#[test]
fn deepseek_v3_manifest_uses_mla_fp8_scale_and_split_expert_names() {
    let metadata = parse_hf_config_metadata(deepseek_v3_config()).unwrap();
    let manifest = build_hf_tensor_manifest(&plan_hf_weight_layout(&metadata).unwrap()).unwrap();

    assert_eq!(manifest.entries.len(), 4737);
    assert_entry(
        &manifest,
        "model.layers.0.self_attn.q_a_proj.weight",
        WeightBlockRole::DeepSeekQALoraProjection,
        1536,
        7168,
        DType::F8E4M3,
    );
    assert_entry(
        &manifest,
        "model.layers.0.self_attn.q_a_proj.weight_scale_inv",
        WeightBlockRole::DeepSeekQALoraScaleInv,
        12,
        56,
        DType::F32,
    );
    assert_entry(
        &manifest,
        "model.layers.0.self_attn.q_a_layernorm.weight",
        WeightBlockRole::DeepSeekQALoraNorm,
        1536,
        1,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "model.layers.3.mlp.gate.e_score_correction_bias",
        WeightBlockRole::RouterCorrectionBias,
        256,
        1,
        DType::F32,
    );
    assert_entry(
        &manifest,
        "model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv",
        WeightBlockRole::ExpertGateScaleInv,
        16,
        56,
        DType::F32,
    );
    assert_entry(
        &manifest,
        "model.layers.3.mlp.shared_experts.down_proj.weight",
        WeightBlockRole::SharedExpertDownProjection,
        7168,
        2048,
        DType::F8E4M3,
    );
}

#[test]
fn deepseek_v32_manifest_adds_indexer_and_f32_norms() {
    let metadata = parse_hf_config_metadata(deepseek_v32_config()).unwrap();
    let manifest = build_hf_tensor_manifest(&plan_hf_weight_layout(&metadata).unwrap()).unwrap();

    assert_eq!(manifest.entries.len(), 1649);
    assert_entry(
        &manifest,
        "model.layers.0.input_layernorm.weight",
        WeightBlockRole::AttentionNorm,
        7168,
        1,
        DType::F32,
    );
    assert_entry(
        &manifest,
        "model.layers.0.self_attn.indexer.wq_b.weight",
        WeightBlockRole::DeepSeekIndexerQueryProjection,
        8192,
        1536,
        DType::F8E4M3,
    );
    assert_entry(
        &manifest,
        "model.layers.0.self_attn.indexer.wq_b.weight_scale_inv",
        WeightBlockRole::DeepSeekIndexerQueryScaleInv,
        64,
        12,
        DType::F32,
    );
    assert_entry(
        &manifest,
        "model.layers.0.self_attn.indexer.weights_proj.weight",
        WeightBlockRole::DeepSeekIndexerWeightsProjection,
        64,
        7168,
        DType::BF16,
    );
}

#[test]
fn deepseek_v4_manifest_covers_mhc_compressors_indexer_hash_and_fp4_experts() {
    let metadata = parse_hf_config_metadata(deepseek_v4_flash_config()).unwrap();
    let manifest = build_hf_tensor_manifest(&plan_hf_weight_layout(&metadata).unwrap()).unwrap();

    assert_entry(
        &manifest,
        "embed.weight",
        WeightBlockRole::TokenEmbedding,
        129280,
        4096,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "hc_head_base",
        WeightBlockRole::DeepSeekV4HcHeadBase,
        4,
        1,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "hc_head_fn",
        WeightBlockRole::DeepSeekV4HcHeadFn,
        4,
        16384,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "hc_head_scale",
        WeightBlockRole::DeepSeekV4HcHeadScale,
        1,
        1,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "layers.0.hc_attn_fn",
        WeightBlockRole::DeepSeekV4HcAttnFn,
        24,
        16384,
        DType::F32,
    );
    assert_entry(
        &manifest,
        "layers.0.attn.attn_sink",
        WeightBlockRole::DeepSeekV4AttentionSink,
        64,
        1,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "layers.0.attn.wq_a.scale",
        WeightBlockRole::DeepSeekV4WqAScale,
        8,
        32,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "layers.2.attn.compressor.ape",
        WeightBlockRole::DeepSeekV4CompressorApe,
        4,
        1024,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "layers.2.attn.indexer.compressor.wkv.weight",
        WeightBlockRole::DeepSeekV4IndexerCompressorWkvProjection,
        256,
        4096,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "layers.2.attn.indexer.weights_proj.weight",
        WeightBlockRole::DeepSeekV4IndexerWeightsProjection,
        64,
        4096,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "layers.3.attn.compressor.ape",
        WeightBlockRole::DeepSeekV4CompressorApe,
        128,
        512,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "layers.0.ffn.gate.tid2eid",
        WeightBlockRole::DeepSeekV4HashRouteTable,
        129280,
        6,
        DType::I64,
    );
    assert_entry(
        &manifest,
        "layers.3.ffn.gate.bias",
        WeightBlockRole::RouterCorrectionBias,
        256,
        1,
        DType::F32,
    );
    assert_entry(
        &manifest,
        "layers.0.ffn.shared_experts.w1.scale",
        WeightBlockRole::DeepSeekV4SharedExpertGateScale,
        2048,
        256,
        DType::F8E4M3,
    );
    assert_entry(
        &manifest,
        "layers.0.ffn.experts.0.w1.weight",
        WeightBlockRole::ExpertGateProjection,
        2048,
        2048,
        DType::U8,
    );
    assert_entry(
        &manifest,
        "layers.0.ffn.experts.0.w1.scale",
        WeightBlockRole::DeepSeekV4ExpertGateScale,
        2048,
        256,
        DType::F8E4M3,
    );
}

#[test]
fn deepseek_v4_bf16_shared_experts_do_not_require_quant_scales() {
    let config = deepseek_v4_flash_config()
        .replace("\"expert_dtype\": \"fp4\"", "\"expert_dtype\": \"bf16\"");
    let metadata = parse_hf_config_metadata(&config).unwrap();
    let manifest = build_hf_tensor_manifest(&plan_hf_weight_layout(&metadata).unwrap()).unwrap();

    assert_entry(
        &manifest,
        "layers.0.ffn.shared_experts.w1.weight",
        WeightBlockRole::SharedExpertGateProjection,
        2048,
        4096,
        DType::BF16,
    );
    assert!(
        !manifest
            .entries
            .iter()
            .any(|entry| entry.name == "layers.0.ffn.shared_experts.w1.scale")
    );
    assert_entry(
        &manifest,
        "layers.0.ffn.experts.0.w1.scale",
        WeightBlockRole::DeepSeekV4ExpertGateScale,
        2048,
        32,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "layers.2.attn.compressor.wkv.weight",
        WeightBlockRole::DeepSeekV4CompressorWkvProjection,
        1024,
        4096,
        DType::F8E4M3,
    );
    assert_entry(
        &manifest,
        "layers.2.attn.compressor.wkv.scale",
        WeightBlockRole::DeepSeekV4CompressorWkvScale,
        8,
        32,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "layers.2.attn.indexer.weights_proj.weight",
        WeightBlockRole::DeepSeekV4IndexerWeightsProjection,
        64,
        4096,
        DType::F8E4M3,
    );
    assert_entry(
        &manifest,
        "layers.2.attn.indexer.weights_proj.scale",
        WeightBlockRole::DeepSeekV4IndexerWeightsScale,
        1,
        32,
        DType::BF16,
    );
    assert_entry(
        &manifest,
        "layers.0.ffn.experts.0.w1.weight",
        WeightBlockRole::ExpertGateProjection,
        2048,
        512,
        DType::I32,
    );
}

fn has_role(
    layer: &crate::hf::deepseek_runtime::DeepSeekLayerWeightContract,
    role: WeightBlockRole,
) -> bool {
    layer.roles.contains(&role)
}

fn assert_entry(
    manifest: &crate::weights::manifest::HfTensorManifest,
    name: &str,
    role: WeightBlockRole,
    rows: usize,
    cols: usize,
    dtype: DType,
) {
    let entry = manifest
        .entries
        .iter()
        .find(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("missing manifest entry {name}"));
    assert_eq!(entry.role, role);
    assert_eq!(entry.rows, rows);
    assert_eq!(entry.cols, cols);
    assert_eq!(entry.dtype, dtype);
}

fn deepseek_v3_config() -> &'static str {
    r#"{
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 6,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "first_k_dense_replace": 3,
        "moe_layer_freq": 1,
        "n_group": 8,
        "topk_group": 4,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": true,
        "routed_scaling_factor": 2.5,
        "num_nextn_predict_layers": 1,
        "vocab_size": 129280,
        "max_position_embeddings": 163840,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 40.0,
            "original_max_position_embeddings": 4096
        },
        "torch_dtype": "bfloat16"
    }"#
}

fn deepseek_v32_config() -> &'static str {
    r#"{
        "architectures": ["DeepseekV32ForCausalLM"],
        "model_type": "deepseek_v32",
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "first_k_dense_replace": 3,
        "moe_layer_freq": 1,
        "n_group": 8,
        "topk_group": 4,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "norm_topk_prob": true,
        "routed_scaling_factor": 2.5,
        "index_topk": 2048,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "vocab_size": 129280,
        "torch_dtype": "bfloat16"
    }"#
}

fn deepseek_v4_flash_config() -> &'static str {
    r#"{
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "topk_method": "noaux_tc",
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": true,
        "routed_scaling_factor": 1.5,
        "index_topk": 512,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "sliding_window": 4096,
        "compress_ratios": [0, 0, 4, 128],
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 0.000001,
        "num_hash_layers": 3,
        "swiglu_limit": 10.0,
        "expert_dtype": "fp4",
        "vocab_size": 129280,
        "max_position_embeddings": 1048576,
        "compress_rope_theta": 1000000.0,
        "rope_parameters": {
            "rope_type": "deepseek_yarn",
            "rope_theta": 10000.0
        },
        "torch_dtype": "bfloat16"
    }"#
}
