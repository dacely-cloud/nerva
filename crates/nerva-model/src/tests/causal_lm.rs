use crate::causal_lm::smoke::hf_causal_lm_safetensors_smoke;
use crate::causal_lm::summary::HfCausalLmSmokeStatus;
use crate::causal_lm::types::{HfCausalLmContextMode, HfCausalLmDecodeScratch, HfCausalLmModel};
use crate::tests::support::{remove_hf_checkpoint_dir, write_hf_checkpoint_dir};
use crate::weights::layout::entry::WeightBlockRole;
use nerva_core::types::id::token::TokenId;

#[test]
fn hf_causal_lm_loads_safetensors_and_decodes_greedily() {
    let summary = hf_causal_lm_safetensors_smoke(4).unwrap();

    assert_eq!(summary.status, HfCausalLmSmokeStatus::Ok);
    assert!(summary.passed());
    assert_eq!(summary.layers, 1);
    assert_eq!(summary.hidden, 2);
    assert_eq!(summary.vocab_size, 4);
    assert_eq!(summary.manifest_entries, 12);
    assert_eq!(summary.shard_plan_entries, 12);
    assert_eq!(summary.tensors_loaded, 12);
    assert!(summary.final_norm_loaded);
    assert!(!summary.tied_lm_head);
    assert!(summary.parity);
    assert_eq!(summary.ledger_count, 4);
    assert_eq!(summary.cpu_events, 4);
    assert_eq!(summary.execution_decisions, 4);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_ne!(summary.output_hash, 0);
    assert_ne!(summary.data_hash, 0);
    assert!(summary.to_json().contains("\"final_norm_loaded\":true"));
}

#[test]
fn hf_causal_lm_prompt_decode_without_context_uses_seed_only_mode() {
    let dir = write_hf_checkpoint_dir("nerva-hf-causal-lm-prompt", fixture_config());
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let model = loaded.model;
    let mut scratch =
        HfCausalLmDecodeScratch::new(model.shape(), model.metadata().vocab_size).unwrap();
    let prompt = [TokenId(1), TokenId(2)];
    let output = model
        .decode_greedy_from_prompt_tokens(&prompt, 2, &mut scratch)
        .unwrap();

    assert_eq!(
        output.context_mode,
        HfCausalLmContextMode::LastTokenSeedOnly
    );
    assert_eq!(output.context_mode.as_str(), "last_token_seed_only");
    assert_eq!(output.prompt_tokens, prompt);
    assert_eq!(output.seed_token, TokenId(2));
    assert_eq!(output.generated_tokens.len(), 2);
    assert_eq!(output.ledgers.len(), 2);

    remove_hf_checkpoint_dir(&dir);
}

#[test]
fn hf_causal_lm_prompt_decode_with_context_uses_prefill_kv_mode() {
    let dir = write_hf_checkpoint_dir("nerva-hf-causal-lm-context-prompt", fixture_config());
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let model = loaded.model;
    let mut scratch = HfCausalLmDecodeScratch::new_with_context(
        model.shape(),
        model.metadata().vocab_size,
        model.layer_count(),
        4,
    )
    .unwrap();
    let prompt = [TokenId(1), TokenId(2)];
    let output = model
        .decode_greedy_from_prompt_tokens(&prompt, 2, &mut scratch)
        .unwrap();

    assert_eq!(
        output.context_mode,
        HfCausalLmContextMode::PromptPrefillKvDecode
    );
    assert_eq!(output.context_mode.as_str(), "prompt_prefill_kv_decode");
    assert_eq!(output.prompt_tokens, prompt);
    assert_eq!(output.seed_token, TokenId(2));
    assert_eq!(output.generated_tokens.len(), 2);
    assert_eq!(output.ledgers.len(), 2);

    remove_hf_checkpoint_dir(&dir);
}

#[test]
fn hf_causal_lm_loader_accepts_grouped_query_kv_tensors() {
    let dir = write_hf_checkpoint_dir("nerva-hf-causal-lm-gqa", gqa_fixture_config());
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let model = loaded.model;

    assert_eq!(model.shape().hidden, 4);
    assert_eq!(model.shape().heads, 2);
    assert_eq!(model.shape().kv_heads, 1);
    assert_eq!(loaded.summary.manifest.entries.len(), 12);
    assert_eq!(
        loaded.summary.bytes_loaded,
        loaded.summary.manifest.total_weight_bytes
    );
    let key_entry = loaded
        .summary
        .manifest
        .entries
        .iter()
        .find(|entry| entry.role == WeightBlockRole::KeyProjection)
        .unwrap();
    assert_eq!(key_entry.rows, 2);
    assert_eq!(key_entry.cols, 4);
    let mut scratch = HfCausalLmDecodeScratch::new_with_context(
        model.shape(),
        model.metadata().vocab_size,
        model.layer_count(),
        2,
    )
    .unwrap();
    let output = model
        .decode_greedy_from_prompt_tokens(&[TokenId(0)], 1, &mut scratch)
        .unwrap();
    assert_eq!(
        output.context_mode,
        HfCausalLmContextMode::PromptPrefillKvDecode
    );
    assert_eq!(output.generated_tokens, [TokenId(0)]);

    remove_hf_checkpoint_dir(&dir);
}

#[test]
fn hf_causal_lm_prompt_decode_rejects_empty_prompt() {
    let dir = write_hf_checkpoint_dir("nerva-hf-causal-lm-empty-prompt", fixture_config());
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let model = loaded.model;
    let mut scratch =
        HfCausalLmDecodeScratch::new(model.shape(), model.metadata().vocab_size).unwrap();
    let err = model
        .decode_greedy_from_prompt_tokens(&[], 1, &mut scratch)
        .unwrap_err();

    assert!(format!("{err:?}").contains("at least one prompt token"));

    remove_hf_checkpoint_dir(&dir);
}

#[test]
fn hf_causal_lm_prompt_decode_rejects_any_out_of_vocab_token() {
    let dir = write_hf_checkpoint_dir("nerva-hf-causal-lm-bad-prompt", fixture_config());
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let model = loaded.model;
    let mut scratch =
        HfCausalLmDecodeScratch::new(model.shape(), model.metadata().vocab_size).unwrap();
    let err = model
        .decode_greedy_from_prompt_tokens(&[TokenId(99), TokenId(0)], 1, &mut scratch)
        .unwrap_err();

    assert!(format!("{err:?}").contains("token id 99 is outside model vocabulary 4"));

    remove_hf_checkpoint_dir(&dir);
}

fn fixture_config() -> &'static str {
    r#"{
        "model_type": "llama",
        "hidden_size": 2,
        "intermediate_size": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 4,
        "torch_dtype": "float16"
    }"#
}

fn gqa_fixture_config() -> &'static str {
    r#"{
        "model_type": "llama",
        "hidden_size": 4,
        "intermediate_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 4,
        "torch_dtype": "float16"
    }"#
}
