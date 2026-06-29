use crate::hf::architecture::HfArchitectureKind;
use crate::hf::parser::parse_hf_config_metadata;
use crate::weights::layout::entry::WeightBlockRole;
use crate::weights::layout::plan::plan_hf_weight_layout;
use crate::weights::manifest::build_hf_tensor_manifest;

#[test]
fn exact_runtime_contract_accepts_silu_without_biases() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "qwen2",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 16,
                "hidden_act": "silu",
                "attention_bias": false,
                "mlp_bias": false,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();

    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen2);
    assert_eq!(metadata.hidden_act.as_deref(), Some("silu"));
    assert!(!metadata.attention_bias);
    assert!(!metadata.mlp_bias);
    assert!(plan_hf_weight_layout(&metadata).is_ok());
}

#[test]
fn exact_runtime_contract_rejects_unsupported_activation() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 16,
                "hidden_activation": "gelu_pytorch_tanh",
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();

    assert!(plan_hf_weight_layout(&metadata).is_err());
}

#[test]
fn exact_runtime_contract_supports_attention_bias_and_rejects_mlp_bias() {
    let qkv_bias = parse_hf_config_metadata(
        r#"{
                "model_type": "qwen2",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 16,
                "qkv_bias": true,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();
    let attention_bias = parse_hf_config_metadata(
        r#"{
                "model_type": "qwen2",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 16,
                "attention_bias": true,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();
    let mlp_bias = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 16,
                "mlp_bias": true,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();

    assert!(qkv_bias.attention_bias);
    assert!(qkv_bias.attention_qkv_bias);
    assert!(!qkv_bias.attention_output_bias);
    let plan = plan_hf_weight_layout(&qkv_bias).unwrap();
    assert_eq!(
        plan.blocks
            .iter()
            .filter(|block| matches!(
                block.role,
                WeightBlockRole::QueryBias
                    | WeightBlockRole::KeyBias
                    | WeightBlockRole::ValueBias
                    | WeightBlockRole::OutputBias
            ))
            .count(),
        3
    );
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    assert!(
        manifest
            .entries
            .iter()
            .any(|entry| entry.name == "model.layers.0.self_attn.q_proj.bias")
    );
    assert!(
        !manifest
            .entries
            .iter()
            .any(|entry| entry.name == "model.layers.0.self_attn.o_proj.bias")
    );

    assert!(attention_bias.attention_bias);
    let plan = plan_hf_weight_layout(&attention_bias).unwrap();
    assert_eq!(
        plan.blocks
            .iter()
            .filter(|block| matches!(
                block.role,
                WeightBlockRole::QueryBias
                    | WeightBlockRole::KeyBias
                    | WeightBlockRole::ValueBias
                    | WeightBlockRole::OutputBias
            ))
            .count(),
        4
    );
    let manifest = build_hf_tensor_manifest(&plan).unwrap();
    assert!(
        manifest
            .entries
            .iter()
            .any(|entry| entry.name == "model.layers.0.self_attn.q_proj.bias")
    );
    assert!(mlp_bias.mlp_bias);
    assert!(plan_hf_weight_layout(&mlp_bias).is_err());
}
