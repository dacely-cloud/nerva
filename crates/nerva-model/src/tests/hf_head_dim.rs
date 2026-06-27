use crate::hf::parser::parse_hf_config_metadata;
use crate::weights::layout::entry::WeightBlockRole;
use crate::weights::layout::plan::plan_hf_weight_layout;

#[test]
fn explicit_hf_head_dim_drives_attention_projection_width() {
    let metadata = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 5,
                "intermediate_size": 7,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 11,
                "torch_dtype": "float16"
            }"#,
    )
    .unwrap();
    let shape = metadata.block_shape();
    let plan = plan_hf_weight_layout(&metadata).unwrap();

    assert_eq!(metadata.head_dim(), 4);
    assert_eq!(metadata.attention_hidden(), 8);
    assert_eq!(metadata.kv_hidden(), 4);
    assert_eq!(shape.hidden, 5);
    assert_eq!(shape.attention_hidden(), 8);
    assert_eq!(shape.kv_hidden(), 4);
    assert!(metadata.to_json().contains("\"attention_hidden_size\":8"));
    assert!(plan.to_json().contains("\"attention_hidden_size\":8"));

    assert_eq!(plan.blocks[2].role, WeightBlockRole::QueryProjection);
    assert_eq!(plan.blocks[2].rows, 8);
    assert_eq!(plan.blocks[2].cols, 5);
    assert_eq!(plan.blocks[3].role, WeightBlockRole::KeyProjection);
    assert_eq!(plan.blocks[3].rows, 4);
    assert_eq!(plan.blocks[3].cols, 5);
    assert_eq!(plan.blocks[5].role, WeightBlockRole::OutputProjection);
    assert_eq!(plan.blocks[5].rows, 5);
    assert_eq!(plan.blocks[5].cols, 8);
}

#[test]
fn derived_hf_head_dim_still_rejects_non_divisible_hidden_size() {
    let parsed = parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 5,
                "intermediate_size": 7,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 11,
                "torch_dtype": "float16"
            }"#,
    );

    assert!(parsed.is_err());
}
