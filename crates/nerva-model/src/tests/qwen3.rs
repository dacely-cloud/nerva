use crate::hf::architecture::HfArchitectureKind;
use crate::hf::parser::parse_hf_config_metadata;
use crate::weights::layout::entry::WeightBlockRole;
use crate::weights::layout::plan::plan_hf_weight_layout;
use crate::weights::manifest::build_hf_tensor_manifest;

#[test]
fn qwen3_dense_config_requires_qk_norm_tensors() {
    let metadata = parse_hf_config_metadata(qwen3_dense_config()).unwrap();
    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen3);
    assert!(metadata.qk_norm);
    assert!(metadata.to_json().contains("\"qk_norm\":true"));

    let plan = plan_hf_weight_layout(&metadata).unwrap();
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    assert_eq!(plan.blocks.len(), 14);
    assert!(manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::QueryNorm
            && entry.name == "model.layers.0.self_attn.q_norm.weight"
            && entry.rows == metadata.head_dim
    }));
    assert!(manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::KeyNorm
            && entry.name == "model.layers.0.self_attn.k_norm.weight"
            && entry.rows == metadata.head_dim
    }));
}

#[test]
fn qwen3_variants_remain_outside_dense_exact_contract() {
    let metadata = parse_hf_config_metadata(
        r#"{
            "model_type": "qwen3_moe",
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 16,
            "torch_dtype": "float16"
        }"#,
    )
    .unwrap();

    assert_eq!(metadata.architecture, HfArchitectureKind::Unknown);
    assert!(plan_hf_weight_layout(&metadata).is_err());
}

fn qwen3_dense_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 16,
        "hidden_act": "silu",
        "rope_theta": 1000000.0,
        "rms_norm_eps": 0.000001,
        "attention_bias": false,
        "mlp_bias": false,
        "torch_dtype": "bfloat16"
    }"#
}
