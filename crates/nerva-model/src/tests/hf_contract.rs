use crate::hf::architecture::HfArchitectureKind;
use crate::hf::parser::parse_hf_config_metadata;
use crate::weights::layout::plan::plan_hf_weight_layout;

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
fn exact_runtime_contract_rejects_attention_and_mlp_bias() {
    let attention_bias = parse_hf_config_metadata(
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

    assert!(attention_bias.attention_bias);
    assert!(plan_hf_weight_layout(&attention_bias).is_err());
    assert!(mlp_bias.mlp_bias);
    assert!(plan_hf_weight_layout(&mlp_bias).is_err());
}
