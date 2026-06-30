use crate::common::shape::TransformerBlockShape;
use crate::hf::architecture::HfArchitectureKind;
use crate::hf::parser::parse_hf_config_metadata;
use crate::hf::probe::{HfMetadataProbeStatus, hf_metadata_probe};
use nerva_core::types::dtype::DType;

#[test]
fn parses_llama_hf_config_metadata() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "rms_norm_eps": 0.000001,
                "rope_theta": 10000.0,
                "bos_token_id": 1,
                "eos_token_id": [2, 3],
                "torch_dtype": "bfloat16"
            }"#,
    )
    .unwrap();

    assert_eq!(metadata.architecture, HfArchitectureKind::Llama);
    assert_eq!(
        metadata.block_shape(),
        TransformerBlockShape::new_with_kv_heads(4096, 32, 8, 11008)
    );
    assert_eq!(metadata.head_dim(), 128);
    assert_eq!(metadata.kv_groups(), 4);
    assert_eq!(metadata.torch_dtype, Some(DType::BF16));
    assert_eq!(metadata.bos_token_id, Some(1));
    assert_eq!(metadata.eos_token_id, Some(2));
    assert!(!metadata.tie_word_embeddings);
    assert!(metadata.to_json().contains("\"architecture\":\"llama\""));
    assert!(metadata.to_json().contains("\"eos_token_id\":2"));
}

#[test]
fn parses_model_type_and_defaults_kv_heads_to_attention_heads() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "mistral",
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "vocab_size": 32000,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();

    assert_eq!(metadata.architecture, HfArchitectureKind::Mistral);
    assert_eq!(metadata.num_key_value_heads, 32);
    assert_eq!(metadata.kv_groups(), 1);
    assert_eq!(metadata.torch_dtype, Some(DType::F16));
}

#[test]
fn parses_tied_word_embedding_config() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "qwen2",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 12,
                "tie_word_embeddings": true,
                "torch_dtype": "bfloat16"
            }"#,
    )
    .unwrap();

    assert_eq!(metadata.architecture, HfArchitectureKind::Qwen2);
    assert!(metadata.tie_word_embeddings);
    assert!(metadata.to_json().contains("\"tie_word_embeddings\":true"));
}

#[test]
fn parses_default_rope_parameters_object() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "qwen2",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 12,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 1000000.0
                }
            }"#,
    )
    .unwrap();

    assert_eq!(metadata.rope_theta, Some(1_000_000.0));
}

#[test]
fn rejects_invalid_hf_metadata_shapes_and_dtypes() {
    let bad_heads = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4097,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000
            }"#,
    );
    let bad_dtype = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "torch_dtype": "float128"
            }"#,
    );
    let bad_tie_word_embeddings = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "tie_word_embeddings": "yes"
            }"#,
    );
    let bad_eos_array = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "eos_token_id": [2 "bad"]
            }"#,
    );
    let unsupported_rope_scaling = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "rope_scaling": {"rope_type": "llama3", "factor": 8.0}
            }"#,
    );

    assert!(bad_heads.is_err());
    assert!(bad_dtype.is_err());
    assert!(bad_tie_word_embeddings.is_err());
    assert!(bad_eos_array.is_err());
    assert!(unsupported_rope_scaling.is_err());
}

#[test]
fn parses_quantized_and_compute_dtype_aliases() {
    for (label, expected) in [
        ("tf32", DType::TF32),
        ("bf32", DType::TF32),
        ("fp8", DType::F8E4M3),
        ("float8_e5m2", DType::F8E5M2),
        ("fp8_e8m0", DType::F8E8M0),
        ("nvfp4", DType::F4E2M1),
        ("mxfp4", DType::F4E2M1),
        ("int4", DType::I4),
        ("uint4", DType::U4),
        ("int8", DType::I8),
    ] {
        let metadata = parse_hf_config_metadata(&format!(
            r#"{{
                "model_type": "llama",
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "vocab_size": 32000,
                "torch_dtype": "{label}"
            }}"#
        ))
        .unwrap();
        assert_eq!(metadata.torch_dtype, Some(expected), "{label}");
    }
}

#[test]
fn hf_metadata_probe_reports_valid_shape() {
    let summary = hf_metadata_probe().unwrap();

    assert_eq!(summary.status, HfMetadataProbeStatus::Ok);
    assert_eq!(summary.metadata.architecture, HfArchitectureKind::Llama);
    assert_eq!(summary.metadata.hidden_size, 4096);
    assert_eq!(summary.metadata.num_attention_heads, 32);
    assert_eq!(summary.metadata.num_key_value_heads, 8);
    assert_eq!(summary.metadata.head_dim(), 128);
    assert_eq!(summary.metadata.kv_groups(), 4);
    assert_ne!(summary.metadata_hash, 0);
    assert!(summary.to_json().contains("\"metadata\""));
}
