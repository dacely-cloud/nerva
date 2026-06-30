use crate::model_io::deepseek::{
    DeepSeekCudaPrimitiveBenchSample, DeepSeekCudaPrimitiveReport,
    deepseek_cuda_primitive_bench_report_json, deepseek_cuda_readiness_report_json,
    run_deepseek_runtime_plan, run_deepseek_vllm_parity_gate, run_deepseek_vllm_reference_audit,
};

#[test]
fn deepseek_v4_runtime_plan_reports_vllm_gap_and_layer_mix() {
    let dir = std::env::temp_dir().join(format!(
        "nerva-bench-deepseek-runtime-plan-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let config_path = dir.join("config.json");
    std::fs::write(&config_path, deepseek_v4_config()).unwrap();

    let json = run_deepseek_runtime_plan(Some(config_path.to_string_lossy().into_owned()))
        .expect("deepseek runtime plan should parse V4 config");

    assert!(json.contains("\"schema\":\"nerva-deepseek-runtime-plan-v1\""));
    assert!(json.contains("\"architecture\":\"deepseek_v4\""));
    assert!(json.contains("\"v4_swa_layers\":2"));
    assert!(json.contains("\"v4_c4_layers\":1"));
    assert!(json.contains("\"v4_c128_layers\":1"));
    assert!(json.contains("\"v4_indexer_layers\":1"));
    assert!(json.contains("\"v4_hash_router_layers\":3"));
    assert!(json.contains("\"runtime_status\":\"unsupported\""));
    assert!(json.contains("\"claim_allowed\":false"));
    assert!(json.contains("fp8_e4m3fn_decode_matches_torch"));
    assert!(json.contains("e8m0_scale_upcast_matches_vllm_raw_exponent_path"));
    assert!(json.contains("cuda_fp8_e4m3fn_e8m0_block_dequant_smoke"));
    assert!(json.contains("deepseek_vllm_kv_cache_spec_planner"));
    assert!(json.contains("cuda_fp8_e4m3fn_e8m0_dequant_api"));
    assert!(json.contains("deepseek_mla_decode_mqa_reference"));
    assert!(json.contains("cuda_deepseek_mla_decode_api"));
    assert!(json.contains("cuda_deepseek_mla_decode_mqa_smoke"));
    assert!(json.contains("deepseek_v3_grouped_sigmoid_router_reference"));
    assert!(json.contains("cuda_deepseek_routed_moe_api"));
    assert!(json.contains("precision_moe_deepseek_v3_grouped_router"));
    assert!(json.contains("precision_moe_deepseek_router_correction_bias_load"));
    assert!(json.contains("cuda_deepseek_router_route_api"));
    assert!(json.contains("cuda_deepseek_v3_grouped_sigmoid_router_smoke"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v3_grouped_router_runtime"));
    assert!(json.contains("deepseek_v4_mhc_compressor_indexer_manifest"));
    assert!(json.contains("mxfp4_e2m1_e8m0_block_dequant_reference"));
    assert!(json.contains("cuda_mxfp4_e2m1_e8m0_dequant_api"));
    assert!(json.contains("cuda_mxfp4_e2m1_e8m0_block_dequant_smoke"));
    assert!(json.contains("cuda_deepseek_fused_inv_rope_fp8_quant_api"));
    assert!(json.contains("cuda_deepseek_fused_inv_rope_fp8_quant_smoke"));
    assert!(json.contains("deepseek_v4_sqrtsoftplus_hash_router_reference"));
    assert!(json.contains("precision_moe_deepseek_v4_sqrtsoftplus_router"));
    assert!(json.contains("deepseek_v4_hash_route_table_i64_loader"));
    assert!(json.contains("precision_moe_deepseek_v4_hash_route_table"));
    assert!(json.contains("cuda_deepseek_v4_sqrtsoftplus_hash_router_smoke"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v4_bias_router_runtime"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v4_hash_router_runtime"));
    assert!(json.contains("cuda_deepseek_qkv_rmsnorm_api"));
    assert!(json.contains("cuda_deepseek_qkv_rmsnorm_smoke"));
    assert!(json.contains("cuda_deepseek_fp8_ds_mla_kv_pack_api"));
    assert!(json.contains("cuda_deepseek_fp8_ds_mla_kv_pack_smoke"));
    assert!(json.contains("cuda_deepseek_compressed_slot_mapping_api"));
    assert!(json.contains("cuda_deepseek_compressed_slot_mapping_smoke"));
    assert!(json.contains("cuda_deepseek_c128_topk_metadata_api"));
    assert!(json.contains("cuda_deepseek_c128_topk_metadata_smoke"));
    assert!(json.contains("cuda_deepseek_c4_indexer_topk_api"));
    assert!(json.contains("cuda_deepseek_c4_indexer_topk_smoke"));
    assert!(json.contains("cuda_deepseek_save_partial_states_api"));
    assert!(json.contains("cuda_deepseek_save_partial_states_smoke"));
    assert!(json.contains("cuda_deepseek_compress_norm_rope_fp8_cache_api"));
    assert!(json.contains("cuda_deepseek_compress_norm_rope_fp8_cache_smoke"));
    assert!(json.contains("cuda_deepseek_compress_norm_rope_mxfp4_cache_api"));
    assert!(json.contains("cuda_deepseek_compress_norm_rope_mxfp4_cache_smoke"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v4_index_topk_descriptor"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v4_compressed_scan_metrics"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v4_c4_sparse_topk_runtime"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v4_c4_topk_cover_all_shortcut"));
    assert!(json.contains("cuda_hf_sequence_deepseek_descriptor_abi"));
    assert!(json.contains("cuda_hf_sequence_deepseek_footprint_accounting"));
    assert!(json.contains("cuda_hf_sequence_deepseek_native_layout_pack"));
    assert!(json.contains("cuda_hf_sequence_deepseek_execution_guard"));
    assert!(json.contains("deepseek_v4_mhc_pre_post_head"));
    assert!(json.contains("\"execution_unit_status\""));
    assert!(json.contains("\"unit\":\"deepseek_v4_megamoe_int8_fp4_experts\""));
    assert!(json.contains("implement V4 MegaMoE int8/fp4 expert kernels"));
    assert!(json.contains("/root/vllm/vllm/models/deepseek_v4/attention.py"));

    let _ = std::fs::remove_file(config_path);
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn deepseek_v32_runtime_plan_reports_sparse_indexer_requirement() {
    let dir = std::env::temp_dir().join(format!(
        "nerva-bench-deepseek-v32-runtime-plan-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let config_path = dir.join("config.json");
    std::fs::write(&config_path, deepseek_v32_config()).unwrap();

    let json = run_deepseek_runtime_plan(Some(config_path.to_string_lossy().into_owned()))
        .expect("deepseek runtime plan should parse V3.2 config");

    assert!(json.contains("\"architecture\":\"deepseek_v3.2\""));
    assert!(json.contains("\"moe_layers\":1"));
    assert!(json.contains("\"dense_mlp_layers\":3"));
    assert!(json.contains("deepseek_v32_sparse_attention_indexer"));
    assert!(json.contains("fp8_e4m3fn_e8m0_block_dequant_reference"));
    assert!(json.contains("cuda_fp8_e4m3fn_e8m0_block_dequant_smoke"));
    assert!(json.contains("deepseek_vllm_kv_cache_spec_planner"));
    assert!(json.contains("deepseek_mla_decode_mqa_reference"));
    assert!(json.contains("cuda_deepseek_mla_decode_mqa_smoke"));
    assert!(json.contains("deepseek_v3_grouped_sigmoid_router_reference"));
    assert!(json.contains("precision_moe_deepseek_v3_grouped_router"));
    assert!(json.contains("cuda_deepseek_router_route_api"));
    assert!(json.contains("cuda_deepseek_v3_grouped_sigmoid_router_smoke"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v3_grouped_router_runtime"));
    assert!(json.contains("cuda_deepseek_compressed_slot_mapping_api"));
    assert!(json.contains("cuda_hf_sequence_deepseek_native_layout_pack"));
    assert!(json.contains("\"unit\":\"deepseek_v32_sparse_attention_indexer\""));
    assert!(
        json.contains("consume packed V3.2 sparse indexer query/key/weights offsets in runtime")
    );
    assert!(json.contains("\"runtime_status\":\"unsupported\""));
    assert!(json.contains("\"claim_allowed\":false"));

    let _ = std::fs::remove_file(config_path);
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn deepseek_cuda_readiness_reports_smokes_and_runtime_gaps() {
    let dir = std::env::temp_dir().join(format!(
        "nerva-bench-deepseek-cuda-readiness-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let config_path = dir.join("config.json");
    std::fs::write(&config_path, deepseek_v4_config()).unwrap();
    let primitives = [
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_mla_decode_mqa_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"mismatches\":0}",
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_quant_block_dequant_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"fp8_mismatches\":0,\"mxfp4_mismatches\":0}",
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_fused_inv_rope_fp8_quant_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"fp8_output_hash\":1,\"scale_output_hash\":2,\"packed_scale_output_hash\":3}",
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_routed_moe_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"mismatches\":0}",
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_router_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"v4_hash_mismatches\":0}",
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_qkv_rmsnorm_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"num_tokens\":2,\"q_size\":4,\"kv_size\":3}",
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_fp8_ds_mla_kv_pack_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"token_stride\":576,\"block_bytes\":2336}",
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_compressed_slot_mapping_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"valid_slots\":2,\"pad_slots\":7}",
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_c128_topk_metadata_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"decode_entries\":1,\"prefill_entries\":7}",
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_c4_indexer_topk_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"selected_entries\":4}",
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_save_partial_states_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"written_tokens\":2,\"skipped_tokens\":1}",
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_compress_norm_rope_fp8_cache_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"written_tokens\":2,\"skipped_tokens\":0,\"scale_format\":0}",
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_compress_norm_rope_mxfp4_cache_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"written_tokens\":2,\"skipped_tokens\":0,\"scale_format\":2}",
        },
    ];

    let json = deepseek_cuda_readiness_report_json(
        Some(config_path.to_string_lossy().into_owned()),
        &primitives,
    )
    .expect("deepseek CUDA readiness should parse V4 config");

    assert!(json.contains("\"schema\":\"nerva-deepseek-cuda-readiness-v1\""));
    assert!(json.contains("\"status\":\"primitive_smokes_ok\""));
    assert!(json.contains("\"architecture\":\"deepseek_v4\""));
    assert!(json.contains("\"primitive_status\":\"ok\""));
    assert!(json.contains("\"primitive_smokes_passed\":13"));
    assert!(json.contains("\"primitive_smokes_total\":13"));
    assert!(json.contains("\"cuda_deepseek_mla_decode_mqa_smoke\""));
    assert!(json.contains("\"cuda_deepseek_quant_block_dequant_smoke\""));
    assert!(json.contains("\"cuda_deepseek_fused_inv_rope_fp8_quant_smoke\""));
    assert!(json.contains("\"cuda_deepseek_routed_moe_smoke\""));
    assert!(json.contains("\"cuda_deepseek_router_smoke\""));
    assert!(json.contains("\"cuda_deepseek_qkv_rmsnorm_smoke\""));
    assert!(json.contains("\"cuda_deepseek_fp8_ds_mla_kv_pack_smoke\""));
    assert!(json.contains("\"cuda_deepseek_compressed_slot_mapping_smoke\""));
    assert!(json.contains("\"cuda_deepseek_c128_topk_metadata_smoke\""));
    assert!(json.contains("\"cuda_deepseek_c4_indexer_topk_smoke\""));
    assert!(json.contains("\"cuda_deepseek_save_partial_states_smoke\""));
    assert!(json.contains("\"cuda_deepseek_compress_norm_rope_fp8_cache_smoke\""));
    assert!(json.contains("\"cuda_deepseek_compress_norm_rope_mxfp4_cache_smoke\""));
    assert!(json.contains("\"vllm_kv_cache_plan\""));
    assert!(json.contains("\"execution_unit_status\""));
    assert!(json.contains("\"unit\":\"deepseek_v4_hash_and_bias_router\""));
    assert!(json.contains("\"status\":\"partial\""));
    assert!(json.contains("precision_moe_deepseek_v4_hash_route_table"));
    assert!(json.contains("deepseek_v4_hash_route_table_i64_loader"));
    assert!(json.contains("cuda_deepseek_router_route_api"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v4_bias_router_runtime"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v4_hash_router_runtime"));
    assert!(json.contains("\"unit\":\"deepseek_v4_parallel_attention_gemm_streams\""));
    assert!(json.contains("\"status\":\"missing\""));
    assert!(json.contains("\"default_block_size\":256"));
    assert!(json.contains("\"v4_swa\""));
    assert!(json.contains("\"v4_c4_mla\""));
    assert!(json.contains("\"v4_c128_mla\""));
    assert!(json.contains("\"cache_dtype_str\":\"fp8_ds_mla\""));
    assert!(json.contains("\"page_size_bytes\":1728"));
    assert!(json.contains("cuda_deepseek_fp8_ds_mla_kv_pack_api"));
    assert!(json.contains("cuda_deepseek_qkv_rmsnorm_api"));
    assert!(json.contains("cuda_deepseek_fused_inv_rope_fp8_quant_api"));
    assert!(json.contains("cuda_deepseek_compressed_slot_mapping_api"));
    assert!(json.contains("cuda_deepseek_c128_topk_metadata_api"));
    assert!(json.contains("cuda_deepseek_c4_indexer_topk_api"));
    assert!(json.contains("cuda_deepseek_save_partial_states_api"));
    assert!(json.contains("cuda_deepseek_compress_norm_rope_fp8_cache_api"));
    assert!(json.contains("cuda_deepseek_compress_norm_rope_mxfp4_cache_api"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v4_index_topk_descriptor"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v4_compressed_scan_metrics"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v4_c4_sparse_topk_runtime"));
    assert!(json.contains("cuda_hf_sequence_deepseek_v4_c4_topk_cover_all_shortcut"));
    assert!(json.contains("deepseek_v4_megamoe_int8_fp4_experts"));
    assert!(json.contains("cuda_hf_sequence_deepseek_descriptor_abi"));
    assert!(json.contains("cuda_hf_sequence_deepseek_footprint_accounting"));
    assert!(json.contains("cuda_hf_sequence_deepseek_native_layout_pack"));
    assert!(json.contains("cuda_hf_sequence_deepseek_execution_guard"));
    assert!(json.contains("\"runtime_parity_status\":\"not_verified\""));
    assert!(json.contains("\"performance_status\":\"not_benchmarked\""));
    assert!(json.contains("\"claim_allowed\":false"));
    assert!(json.contains("/root/vllm/vllm/models/deepseek_v4/attention.py"));

    let _ = std::fs::remove_file(config_path);
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn deepseek_cuda_readiness_without_config_reports_unknown_architecture() {
    let primitives = [DeepSeekCudaPrimitiveReport {
        name: "cuda_deepseek_mla_decode_mqa_smoke",
        status: "unavailable",
        summary_json: "{\"status\":\"unavailable\"}",
    }];

    let json = deepseek_cuda_readiness_report_json(None, &primitives)
        .expect("readiness without config should still report smoke status");

    assert!(json.contains("\"status\":\"primitive_smokes_incomplete\""));
    assert!(json.contains("\"architecture\":null"));
    assert!(json.contains("\"primitive_status\":\"unavailable\""));
    assert!(json.contains("\"primitive_smokes_passed\":0"));
    assert!(json.contains("\"primitive_smokes_total\":1"));
    assert!(json.contains("\"implemented_primitives\":[]"));
    assert!(json.contains("\"required_execution_units\":[]"));
    assert!(json.contains("\"vllm_reference_units\":[]"));
    assert!(json.contains("\"vllm_kv_cache_plan\":null"));
    assert!(json.contains("\"execution_unit_status\":[]"));
    assert!(json.contains("\"claim_allowed\":false"));
}

#[test]
fn deepseek_cuda_primitive_bench_report_is_not_end_to_end_claim() {
    let samples = [
        DeepSeekCudaPrimitiveBenchSample {
            name: "router_v3_grouped_sigmoid".to_string(),
            status: "ok",
            requested_iterations: 16,
            executed_iterations: 16,
            total_wall_ns: 1600,
            avg_wall_ns: 100,
            output_hash: 11,
            device_arena_bytes: 128,
            pinned_host_bytes: 96,
            h2d_bytes_per_iter: 64,
            d2h_bytes_per_iter: 32,
            kernel_launches_per_iter: 1,
            sync_calls_per_iter: 1,
            hot_path_allocations_per_iter: 0,
            error: None,
        },
        DeepSeekCudaPrimitiveBenchSample {
            name: "routed_moe_forward".to_string(),
            status: "unavailable",
            requested_iterations: 16,
            executed_iterations: 1,
            total_wall_ns: 500,
            avg_wall_ns: 500,
            output_hash: 0,
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            h2d_bytes_per_iter: 0,
            d2h_bytes_per_iter: 0,
            kernel_launches_per_iter: 0,
            sync_calls_per_iter: 0,
            hot_path_allocations_per_iter: 0,
            error: Some("no CUDA device".to_string()),
        },
    ];

    let json = deepseek_cuda_primitive_bench_report_json(16, &samples);

    assert!(json.contains("\"schema\":\"nerva-deepseek-cuda-primitive-bench-v1\""));
    assert!(json.contains("\"status\":\"unavailable\""));
    assert!(json.contains("\"primitive_samples_total\":2"));
    assert!(json.contains("\"primitive_samples_ok\":1"));
    assert!(json.contains("\"primitive_samples_unavailable\":1"));
    assert!(json.contains("\"runtime_parity_status\":\"primitive_microbench_only\""));
    assert!(json.contains("\"performance_status\":\"not_vllm_end_to_end_comparable\""));
    assert!(json.contains("\"claim_allowed\":false"));
    assert!(json.contains("\"name\":\"router_v3_grouped_sigmoid\""));
    assert!(json.contains("\"avg_wall_ns\":100"));
    assert!(json.contains("\"kernel_launches_per_iter\":1"));
    assert!(json.contains("\"hot_path_allocations_per_iter\":0"));
    assert!(json.contains("/root/vllm/vllm/model_executor/models/deepseek_v2.py"));
    assert!(json.contains("/root/vllm/vllm/models/deepseek_v4/nvidia/model.py"));
    assert!(json.contains("/root/vllm/vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py"));
}

#[test]
fn deepseek_vllm_reference_audit_pins_expected_source_units() {
    let dir = std::env::temp_dir().join(format!(
        "nerva-bench-deepseek-vllm-audit-{}",
        std::process::id()
    ));
    let root = dir.join("vllm-root");
    write_vllm_reference_fixture(&root);

    let json = run_deepseek_vllm_reference_audit(Some(root.to_string_lossy().into_owned()))
        .expect("vLLM reference audit should scan fixture root");

    assert!(json.contains("\"schema\":\"nerva-deepseek-vllm-reference-audit-v1\""));
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"reference_units_total\":14"));
    assert!(json.contains("\"reference_units_ok\":14"));
    assert!(json.contains("\"reference_units_missing_file\":0"));
    assert!(json.contains("\"reference_units_symbol_gap\":0"));
    assert!(json.contains("\"runtime_parity_status\":\"vllm_reference_sources_pinned\""));
    assert!(json.contains("\"performance_status\":\"source_audit_only_not_runtime_benchmark\""));
    assert!(json.contains("\"claim_allowed\":false"));
    assert!(json.contains("\"execution_unit\":\"v3_mla_moe_model\""));
    assert!(json.contains("\"execution_unit\":\"v4_sparse_mla_backend\""));
    assert!(json.contains("\"execution_unit\":\"v4_swa_cache_spec\""));
    assert!(json.contains("\"execution_unit\":\"v4_save_partial_states\""));
    assert!(json.contains("\"execution_unit\":\"v4_fused_qkv_rmsnorm\""));
    assert!(json.contains("\"execution_unit\":\"v4_fused_inv_rope_fp8_quant\""));
    assert!(json.contains("\"execution_unit\":\"v4_fused_compress_quant_cache\""));
    assert!(json.contains("\"fnv1a64\":\"0x"));
    assert!(json.contains("DeepseekV4FlashMLABackend"));
    assert!(json.contains("\"missing_symbols\":[]"));

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn deepseek_vllm_parity_gate_blocks_until_runtime_units_are_complete() {
    let dir = std::env::temp_dir().join(format!(
        "nerva-bench-deepseek-vllm-gate-{}",
        std::process::id()
    ));
    let vllm_root = dir.join("vllm-root");
    write_vllm_reference_fixture(&vllm_root);
    std::fs::create_dir_all(&dir).unwrap();
    let config_path = dir.join("config.json");
    std::fs::write(&config_path, deepseek_v4_config()).unwrap();

    let json = run_deepseek_vllm_parity_gate(
        Some(config_path.to_string_lossy().into_owned()),
        Some(vllm_root.to_string_lossy().into_owned()),
    )
    .expect("DeepSeek parity gate should parse config and fixture references");

    assert!(json.contains("\"schema\":\"nerva-deepseek-vllm-parity-gate-v1\""));
    assert!(json.contains("\"status\":\"runtime_blocked\""));
    assert!(json.contains("\"architecture\":\"deepseek_v4\""));
    assert!(json.contains("\"runtime_contract_status\":\"unsupported\""));
    assert!(json.contains("\"vllm_reference_status\":\"ok\""));
    assert!(json.contains("\"vllm_reference_units_total\":14"));
    assert!(json.contains("\"vllm_reference_units_ok\":14"));
    assert!(json.contains("\"runtime_units_total\":8"));
    assert!(json.contains("\"runtime_blocking_units_total\":8"));
    assert!(json.contains("\"runtime_units_partial\":7"));
    assert!(json.contains("\"runtime_units_missing\":1"));
    assert!(json.contains("\"deepseek_v4_parallel_attention_gemm_streams\""));
    assert!(json.contains("\"runtime_parity_status\":\"blocked_before_end_to_end_parity\""));
    assert!(json.contains("\"performance_status\":\"blocked_until_runtime_units_complete\""));
    assert!(json.contains("\"claim_allowed\":false"));
    assert!(json.contains("\"performance_comparison_allowed\":false"));
    assert!(json.contains("verify full-layer routed outputs against vLLM"));

    let _ = std::fs::remove_dir_all(dir);
}

fn write_vllm_reference_fixture(root: &std::path::Path) {
    write_fixture_file(
        root,
        "vllm/model_executor/models/deepseek_v2.py",
        r#"
class DeepseekV2MLAAttention: pass
class DeepseekV2MoE: pass
FusedMoE(
MultiHeadLatentAttentionWrapper
MLAAttentionSpec
DeepseekV32IndexerBackend
"#,
    );
    write_fixture_file(
        root,
        "vllm/v1/attention/backends/mla/indexer.py",
        r#"
class DeepseekV32IndexerBackend: pass
class DeepseekV4IndexerBackend: pass
compress_ratio
get_compressed_slot_mapping
DeepseekV32IndexerMetadataBuilder
get_supported_kernel_block_sizes
return [1, 64] if current_platform.is_rocm() else [64]
"#,
    );
    write_fixture_file(
        root,
        "vllm/v1/kv_cache_interface.py",
        r#"
class MLAAttentionSpec: pass
class SlidingWindowMLASpec: pass
fp8_ds_mla
compress_ratio
real_page_size_bytes
return self.block_size // self.compress_ratio
return self.storage_block_size * 584
return self.block_size * 656
_apply_alignment_padding
"#,
    );
    write_fixture_file(
        root,
        "vllm/models/deepseek_v4/attention.py",
        r#"
class DeepseekV4Attention: pass
_resolve_dsv4_kv_cache_dtype
DeepseekCompressor
execute_in_parallel
MLAAttentionSpec
compress_ratio
fp8_ds_mla
DeepseekV4SWACache
alignment=576 if uses_fp8_ds_mla_layout else None
self.quant_block_size = 128
"#,
    );
    write_fixture_file(
        root,
        "vllm/models/deepseek_v4/compressor.py",
        r#"
class DeepseekCompressor: pass
save_partial_states
compress_norm_rope_store_triton
compress_ratio
SlidingWindowMLASpec
alignment=576
CompressorMetadataBuilder
"#,
    );
    write_fixture_file(
        root,
        "vllm/v1/attention/backends/mla/sparse_swa.py",
        r#"
class DeepseekV4SWACache: pass
self.block_size = 64
SlidingWindowMLASpec
alignment=576 if uses_fp8_ds_mla_layout else None
model_version="deepseek_v4"
return (num_blocks, block_size, 584)
"#,
    );
    write_fixture_file(
        root,
        "vllm/models/deepseek_v4/common/ops/save_partial_states.py",
        r#"
def save_partial_states(): pass
_save_partial_states_kernel
slot_id < 0
score + ape
"#,
    );
    write_fixture_file(
        root,
        "vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py",
        r#"
def fused_q_kv_rmsnorm(): pass
_fused_q_kv_rmsnorm_kernel
num_tokens
pid_task
RMSNorm in fp32
"#,
    );
    write_fixture_file(
        root,
        "vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py",
        r#"
def fused_inv_rope_fp8_quant(): pass
_fused_inv_rope_fp8_quant_per_head
TMA_ALIGNED_SCALES
float8e4nv
packed_val
"#,
    );
    write_fixture_file(
        root,
        "vllm/models/deepseek_v4/sparse_mla.py",
        r#"
class DeepseekV4FlashMLABackend: pass
FLASHMLA_SPARSE_DSV4
fp8_ds_mla
584
return [256]
return (num_blocks, block_size, 584)
DeepseekV4FlashMLAMetadataBuilder
build_c128a_topk_metadata
"#,
    );
    write_fixture_file(
        root,
        "vllm/models/deepseek_v4/nvidia/model.py",
        r#"
class DeepseekV4MegaMoEExperts: pass
prepare_megamoe_inputs
fused_topk_bias
class DeepseekV4MoE: pass
class DeepseekV4ForCausalLM: pass
"#,
    );
    write_fixture_file(
        root,
        "vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py",
        r#"
def prepare_megamoe_inputs(): pass
_prepare_megamoe_inputs_kernel
"#,
    );
    write_fixture_file(
        root,
        "vllm/models/deepseek_v4/nvidia/ops/o_proj.py",
        r#"
def compute_fp8_einsum_recipe(): pass
def deep_gemm_fp8_o_proj(): pass
fused_inv_rope_fp8_quant
fp8_einsum
"#,
    );
    write_fixture_file(
        root,
        "vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py",
        r#"
compress_norm_rope_store_triton
_fused_kv_compress_norm_rope_insert_sparse_attn
_fused_kv_compress_norm_rope_insert_indexer_attn
_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
COMPRESS_RATIO
"#,
    );
}

fn write_fixture_file(root: &std::path::Path, relative: &str, content: &str) {
    let path = root.join(relative);
    std::fs::create_dir_all(path.parent().expect("fixture file should have a parent")).unwrap();
    std::fs::write(path, content).unwrap();
}

fn deepseek_v4_config() -> &'static str {
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
