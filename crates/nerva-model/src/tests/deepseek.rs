use crate::hf::architecture::HfArchitectureKind;
use crate::hf::contract::validate_exact_runtime_contract;
use crate::hf::metadata::HfMlpLayerKind;
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
    assert_eq!(metadata.index_n_heads, Some(64));
    assert_eq!(metadata.index_head_dim, Some(128));
    assert_eq!(metadata.mlp_layer_types[0], HfMlpLayerKind::Dense);
    assert_eq!(metadata.mlp_layer_types[3], HfMlpLayerKind::SparseMoe);
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
    assert_eq!(metadata.qk_nope_head_dim, Some(448));
    assert_eq!(metadata.qk_rope_head_dim, Some(64));
    assert_eq!(metadata.v_head_dim, Some(512));
    assert_eq!(metadata.num_experts, Some(256));
    assert_eq!(metadata.num_experts_per_tok, Some(6));
    assert_eq!(metadata.moe_intermediate_size, Some(2048));
    assert_eq!(metadata.shared_expert_intermediate_size, Some(2048));
    assert_eq!(metadata.scoring_func.as_deref(), Some("sqrtsoftplus"));
    assert_eq!(metadata.index_topk, Some(512));
    assert_eq!(metadata.compress_ratios, vec![0, 0, 4, 128]);
    assert_eq!(metadata.hc_mult, Some(4));
    assert_eq!(metadata.hc_sinkhorn_iters, Some(20));
    assert_eq!(metadata.hc_eps, Some(0.000001));
    assert!(
        metadata
            .mlp_layer_types
            .iter()
            .all(|kind| *kind == HfMlpLayerKind::SparseMoe)
    );
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
    }

    assert!(
        plan_hf_weight_layout(&parse_hf_config_metadata(deepseek_v3_config()).unwrap()).is_ok()
    );
    assert!(
        plan_hf_weight_layout(&parse_hf_config_metadata(deepseek_v32_config()).unwrap()).is_ok()
    );
    assert!(
        plan_hf_weight_layout(&parse_hf_config_metadata(deepseek_v4_flash_config()).unwrap())
            .is_err()
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
        "num_hidden_layers": 3,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
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
        "compress_ratios": [0, 0, 4, 128],
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 0.000001,
        "vocab_size": 129280,
        "max_position_embeddings": 1048576,
        "rope_parameters": {
            "rope_type": "deepseek_yarn",
            "rope_theta": 10000.0
        },
        "torch_dtype": "bfloat16"
    }"#
}
