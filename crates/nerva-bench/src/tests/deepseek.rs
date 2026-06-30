use crate::model_io::deepseek::{
    DeepSeekCudaPrimitiveBenchSample, DeepSeekCudaPrimitiveReport,
    deepseek_cuda_primitive_bench_report_json, deepseek_cuda_readiness_report_json,
    run_deepseek_runtime_plan,
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
    assert!(json.contains("deepseek_v4_mhc_compressor_indexer_manifest"));
    assert!(json.contains("mxfp4_e2m1_e8m0_block_dequant_reference"));
    assert!(json.contains("cuda_mxfp4_e2m1_e8m0_dequant_api"));
    assert!(json.contains("cuda_mxfp4_e2m1_e8m0_block_dequant_smoke"));
    assert!(json.contains("deepseek_v4_sqrtsoftplus_hash_router_reference"));
    assert!(json.contains("precision_moe_deepseek_v4_sqrtsoftplus_router"));
    assert!(json.contains("deepseek_v4_hash_route_table_i64_loader"));
    assert!(json.contains("precision_moe_deepseek_v4_hash_route_table"));
    assert!(json.contains("cuda_deepseek_v4_sqrtsoftplus_hash_router_smoke"));
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
    assert!(json.contains("\"unit\":\"deepseek_v32_sparse_attention_indexer\""));
    assert!(json.contains("implement V3.2 sparse indexer query/key/weights runtime"));
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
            name: "cuda_deepseek_routed_moe_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"mismatches\":0}",
        },
        DeepSeekCudaPrimitiveReport {
            name: "cuda_deepseek_router_smoke",
            status: "ok",
            summary_json: "{\"status\":\"ok\",\"v4_hash_mismatches\":0}",
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
    assert!(json.contains("\"primitive_smokes_passed\":4"));
    assert!(json.contains("\"primitive_smokes_total\":4"));
    assert!(json.contains("\"cuda_deepseek_mla_decode_mqa_smoke\""));
    assert!(json.contains("\"cuda_deepseek_quant_block_dequant_smoke\""));
    assert!(json.contains("\"cuda_deepseek_routed_moe_smoke\""));
    assert!(json.contains("\"cuda_deepseek_router_smoke\""));
    assert!(json.contains("\"vllm_kv_cache_plan\""));
    assert!(json.contains("\"execution_unit_status\""));
    assert!(json.contains("\"unit\":\"deepseek_v4_hash_and_bias_router\""));
    assert!(json.contains("\"status\":\"partial\""));
    assert!(json.contains("precision_moe_deepseek_v4_hash_route_table"));
    assert!(json.contains("deepseek_v4_hash_route_table_i64_loader"));
    assert!(json.contains("cuda_deepseek_router_route_api"));
    assert!(json.contains("integrate hash and bias routing into CUDA exact runtime decode layers"));
    assert!(json.contains("\"unit\":\"deepseek_v4_parallel_attention_gemm_streams\""));
    assert!(json.contains("\"status\":\"missing\""));
    assert!(json.contains("\"default_block_size\":256"));
    assert!(json.contains("\"v4_swa\""));
    assert!(json.contains("\"v4_c4_mla\""));
    assert!(json.contains("\"v4_c128_mla\""));
    assert!(json.contains("\"cache_dtype_str\":\"fp8_ds_mla\""));
    assert!(json.contains("\"page_size_bytes\":1728"));
    assert!(json.contains("deepseek_v4_megamoe_int8_fp4_experts"));
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
