use crate::causal_lm::types::HfCausalLmModel;
use crate::tests::support::{remove_hf_checkpoint_dir, write_hf_checkpoint_dir};
use crate::weights::layout::entry::WeightBlockRole;

#[test]
fn hf_causal_lm_loader_reads_attention_bias_tensors() {
    let dir = write_hf_checkpoint_dir("nerva-hf-causal-lm-bias", bias_fixture_config());
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();

    assert!(loaded.model.metadata().attention_bias);
    assert_eq!(loaded.summary.manifest.entries.len(), 16);
    assert_eq!(loaded.summary.shard_plan.entries.len(), 16);
    assert!(
        loaded
            .summary
            .manifest
            .entries
            .iter()
            .any(|entry| entry.role == WeightBlockRole::OutputBias)
    );
    assert_eq!(
        loaded.summary.bytes_loaded,
        loaded.summary.manifest.total_weight_bytes
    );

    remove_hf_checkpoint_dir(&dir);
}

fn bias_fixture_config() -> &'static str {
    r#"{
        "model_type": "qwen2",
        "hidden_size": 2,
        "intermediate_size": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 4,
        "attention_bias": true,
        "torch_dtype": "float16"
    }"#
}
