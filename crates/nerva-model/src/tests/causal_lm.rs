use crate::causal_lm::smoke::hf_causal_lm_safetensors_smoke;
use crate::causal_lm::summary::HfCausalLmSmokeStatus;
use crate::causal_lm::types::{
    HfCausalLmContextMode, HfCausalLmDecodeScratch, HfCausalLmLayer, HfCausalLmModel,
    HfCausalLmStopReason,
};
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
    assert_eq!(output.stop_reason, HfCausalLmStopReason::MaxSteps);
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
    assert_eq!(output.stop_reason, HfCausalLmStopReason::MaxSteps);
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
    assert_eq!(model.layers[0].rope_theta(), Some(10_000.0));
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
    assert_eq!(output.stop_reason, HfCausalLmStopReason::MaxSteps);
    assert_eq!(output.generated_tokens, [TokenId(0)]);

    remove_hf_checkpoint_dir(&dir);
}

#[test]
fn hf_causal_lm_loader_accepts_qwen3_moe_attention_and_decodes() {
    let dir = write_hf_checkpoint_dir("nerva-hf-causal-lm-qwen3-moe", qwen3_moe_config());
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let model = loaded.model;

    assert!(model.metadata().has_moe_layers());
    assert!(model.metadata().qk_norm);
    assert_eq!(model.layer_count(), 1);
    assert!(model.layer(0).is_none());
    assert_eq!(model.layers[0].rope_theta(), Some(10_000.0));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::RouterProjection
            && entry.name == "model.layers.0.mlp.gate.weight"
    }));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::ExpertGateProjection
            && entry.layer == Some(0)
            && entry.expert == Some(0)
            && entry.name == "model.layers.0.mlp.experts.0.gate_proj.weight"
    }));
    assert_eq!(
        loaded.summary.tensors_loaded,
        loaded.summary.manifest.entries.len()
    );

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
    assert_eq!(output.stop_reason, HfCausalLmStopReason::MaxSteps);
    assert_eq!(output.generated_tokens, [TokenId(0)]);

    remove_hf_checkpoint_dir(&dir);
}

#[test]
fn hf_causal_lm_loader_accepts_qwen35_moe_full_attention_and_decodes() {
    let dir = write_hf_checkpoint_dir(
        "nerva-hf-causal-lm-qwen35-moe-full",
        qwen35_moe_full_attention_config(),
    );
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let model = loaded.model;

    assert!(model.metadata().has_moe_layers());
    assert!(!model.metadata().has_linear_attention_layers());
    assert!(model.metadata().qk_norm);
    assert_eq!(model.layer_count(), 1);
    assert!(matches!(
        model.causal_layer(0),
        Some(HfCausalLmLayer::SparseMoe(_))
    ));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::QueryProjection
            && entry.layer == Some(0)
            && entry.name == "model.language_model.layers.0.self_attn.q_proj.weight"
            && entry.rows == 8
            && entry.cols == 4
    }));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::ExpertGateUpProjection
            && entry.layer == Some(0)
            && entry.name == "model.language_model.layers.0.mlp.experts.gate_up_proj"
            && entry.rank == 3
            && entry.depth == Some(4)
            && (entry.rows, entry.cols) == (6, 4)
    }));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::ExpertDownProjection
            && entry.layer == Some(0)
            && entry.name == "model.language_model.layers.0.mlp.experts.down_proj"
            && entry.rank == 3
            && entry.depth == Some(4)
            && (entry.rows, entry.cols) == (4, 3)
    }));
    assert_eq!(
        loaded.summary.tensors_loaded,
        loaded.summary.manifest.entries.len()
    );

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
    assert_eq!(output.stop_reason, HfCausalLmStopReason::MaxSteps);
    assert_eq!(output.generated_tokens, [TokenId(0)]);

    remove_hf_checkpoint_dir(&dir);
}

#[test]
fn hf_causal_lm_loader_constructs_qwen35_moe_gdn_layer_and_decodes() {
    let dir = write_hf_checkpoint_dir(
        "nerva-hf-causal-lm-qwen35-moe-gdn",
        qwen35_moe_linear_attention_config(),
    );
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let model = loaded.model;

    assert!(model.metadata().has_moe_layers());
    assert!(model.metadata().has_linear_attention_layers());
    assert_eq!(model.layer_count(), 1);
    let Some(HfCausalLmLayer::GatedDeltaNetMoe(layer)) = model.causal_layer(0) else {
        panic!("expected Qwen3.5-MoE GatedDeltaNet layer");
    };
    let view = layer.encoded_view();
    assert_eq!(view.gdn.conv_dim().unwrap(), 7);
    assert_eq!(view.linear_conv.len(), 28);
    assert_eq!(view.linear_qkv.len(), 28);
    assert_eq!(view.linear_a_log.len(), 1);
    assert_eq!(view.linear_a_log_bits.len(), 2);
    assert_eq!(view.expert_gate_up.len(), 96);
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::LinearALog
            && entry.layer == Some(0)
            && entry.name == "model.language_model.layers.0.linear_attn.A_log"
    }));
    assert_eq!(
        loaded.summary.tensors_loaded,
        loaded.summary.manifest.entries.len()
    );

    let mut seed_scratch =
        HfCausalLmDecodeScratch::new(model.shape(), model.metadata().vocab_size).unwrap();
    let (tokens, ledgers) = model
        .decode_greedy(TokenId(0), 1, &mut seed_scratch)
        .unwrap();
    assert_eq!(tokens, [TokenId(0)]);
    assert_eq!(ledgers.len(), 1);

    let mut context_scratch = HfCausalLmDecodeScratch::new_with_context(
        model.shape(),
        model.metadata().vocab_size,
        model.layer_count(),
        3,
    )
    .unwrap();
    let output = model
        .decode_greedy_from_prompt_tokens(&[TokenId(0), TokenId(1)], 1, &mut context_scratch)
        .unwrap();
    assert_eq!(
        output.context_mode,
        HfCausalLmContextMode::PromptPrefillKvDecode
    );
    assert_eq!(output.stop_reason, HfCausalLmStopReason::MaxSteps);
    assert_eq!(output.generated_tokens, [TokenId(0)]);

    remove_hf_checkpoint_dir(&dir);
}

#[test]
fn hf_causal_lm_loader_accepts_qwen2_moe_shared_expert_and_decodes() {
    let dir = write_hf_checkpoint_dir("nerva-hf-causal-lm-qwen2-moe", qwen2_moe_config());
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let model = loaded.model;

    assert_eq!(model.metadata().architecture.as_str(), "qwen2_moe");
    assert!(model.metadata().has_moe_layers());
    assert_eq!(model.metadata().shared_expert_intermediate_size, Some(3));
    assert_eq!(model.layer_count(), 1);
    assert!(matches!(
        model.causal_layer(0),
        Some(HfCausalLmLayer::SparseMoe(_))
    ));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::SharedExpertGateProjection
            && entry.layer == Some(0)
            && entry.name == "model.layers.0.mlp.shared_expert.gate_proj.weight"
            && (entry.rows, entry.cols) == (3, 4)
    }));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::SharedExpertUpProjection
            && entry.layer == Some(0)
            && entry.name == "model.layers.0.mlp.shared_expert.up_proj.weight"
            && (entry.rows, entry.cols) == (3, 4)
    }));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::SharedExpertDownProjection
            && entry.layer == Some(0)
            && entry.name == "model.layers.0.mlp.shared_expert.down_proj.weight"
            && (entry.rows, entry.cols) == (4, 3)
    }));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::SharedExpertRouterProjection
            && entry.layer == Some(0)
            && entry.name == "model.layers.0.mlp.shared_expert_gate.weight"
            && (entry.rows, entry.cols) == (1, 4)
    }));

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
    assert_eq!(output.stop_reason, HfCausalLmStopReason::MaxSteps);
    assert_eq!(output.generated_tokens, [TokenId(0)]);

    remove_hf_checkpoint_dir(&dir);
}

#[test]
fn hf_causal_lm_loader_accepts_mixtral_moe_and_decodes() {
    let dir = write_hf_checkpoint_dir("nerva-hf-causal-lm-mixtral-moe", mixtral_moe_config());
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let model = loaded.model;

    assert_eq!(model.metadata().architecture.as_str(), "mixtral_moe");
    assert!(model.metadata().has_moe_layers());
    assert_eq!(model.metadata().num_experts, Some(4));
    assert_eq!(model.metadata().moe_intermediate_size, Some(3));
    assert_eq!(model.layer_count(), 1);
    assert!(matches!(
        model.causal_layer(0),
        Some(HfCausalLmLayer::SparseMoe(_))
    ));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::RouterProjection
            && entry.layer == Some(0)
            && entry.name == "model.layers.0.block_sparse_moe.gate.weight"
            && (entry.rows, entry.cols) == (4, 4)
    }));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::ExpertGateProjection
            && entry.layer == Some(0)
            && entry.expert == Some(0)
            && entry.name == "model.layers.0.block_sparse_moe.experts.0.w1.weight"
            && (entry.rows, entry.cols) == (3, 4)
    }));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::ExpertUpProjection
            && entry.layer == Some(0)
            && entry.expert == Some(0)
            && entry.name == "model.layers.0.block_sparse_moe.experts.0.w3.weight"
            && (entry.rows, entry.cols) == (3, 4)
    }));
    assert!(loaded.summary.manifest.entries.iter().any(|entry| {
        entry.role == WeightBlockRole::ExpertDownProjection
            && entry.layer == Some(0)
            && entry.expert == Some(0)
            && entry.name == "model.layers.0.block_sparse_moe.experts.0.w2.weight"
            && (entry.rows, entry.cols) == (4, 3)
    }));

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
    assert_eq!(output.stop_reason, HfCausalLmStopReason::MaxSteps);
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
        "rope_theta": 10000.0,
        "torch_dtype": "float16"
    }"#
}

fn qwen3_moe_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": 4,
        "intermediate_size": 8,
        "moe_intermediate_size": 3,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "norm_topk_prob": true,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 4,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        "rms_norm_eps": 0.000001,
        "attention_bias": false,
        "mlp_bias": false,
        "torch_dtype": "float16"
    }"#
}

fn qwen35_moe_full_attention_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "attention_bias": false,
            "dtype": "float16",
            "hidden_act": "silu",
            "hidden_size": 4,
            "intermediate_size": 8,
            "layer_types": ["full_attention"],
            "mlp_only_layers": [],
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 3,
            "norm_topk_prob": true,
            "num_attention_heads": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 0.000001,
            "shared_expert_intermediate_size": 0,
            "tie_word_embeddings": false,
            "use_qk_norm": true,
            "vocab_size": 4
        },
        "tie_word_embeddings": false
    }"#
}

fn qwen35_moe_linear_attention_config() -> &'static str {
    r#"{
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "attention_bias": false,
            "dtype": "float16",
            "hidden_act": "silu",
            "hidden_size": 4,
            "intermediate_size": 8,
            "layer_types": ["linear_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 2,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_value_head_dim": 3,
            "mlp_only_layers": [],
            "model_type": "qwen3_5_moe_text",
            "moe_intermediate_size": 3,
            "norm_topk_prob": true,
            "num_attention_heads": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 0.000001,
            "shared_expert_intermediate_size": 0,
            "tie_word_embeddings": false,
            "vocab_size": 4
        },
        "tie_word_embeddings": false
    }"#
}

fn qwen2_moe_config() -> &'static str {
    r#"{
        "architectures": ["Qwen2MoeForCausalLM"],
        "model_type": "qwen2_moe",
        "hidden_size": 4,
        "intermediate_size": 8,
        "moe_intermediate_size": 3,
        "shared_expert_intermediate_size": 3,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "norm_topk_prob": false,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 4,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        "rms_norm_eps": 0.000001,
        "attention_bias": false,
        "mlp_bias": false,
        "torch_dtype": "float16"
    }"#
}

fn mixtral_moe_config() -> &'static str {
    r#"{
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "hidden_size": 4,
        "intermediate_size": 3,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 4,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        "rms_norm_eps": 0.000001,
        "attention_bias": false,
        "mlp_bias": false,
        "torch_dtype": "float16"
    }"#
}
