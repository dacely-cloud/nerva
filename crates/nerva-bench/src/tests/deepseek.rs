use crate::model_io::deepseek::run_deepseek_runtime_plan;

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
    assert!(json.contains("deepseek_mla_decode_mqa_reference"));
    assert!(json.contains("deepseek_v3_grouped_sigmoid_router_reference"));
    assert!(json.contains("deepseek_v4_mhc_compressor_indexer_manifest"));
    assert!(json.contains("mxfp4_e2m1_e8m0_block_dequant_reference"));
    assert!(json.contains("deepseek_v4_sqrtsoftplus_hash_router_reference"));
    assert!(json.contains("deepseek_v4_mhc_pre_post_head"));
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
    assert!(json.contains("deepseek_mla_decode_mqa_reference"));
    assert!(json.contains("deepseek_v3_grouped_sigmoid_router_reference"));
    assert!(json.contains("\"runtime_status\":\"unsupported\""));
    assert!(json.contains("\"claim_allowed\":false"));

    let _ = std::fs::remove_file(config_path);
    let _ = std::fs::remove_dir(dir);
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
