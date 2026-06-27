use nerva_core::types::id::token::TokenId;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::causal_lm::smoke::load_hf_causal_lm_smoke_fixture;
use nerva_model::causal_lm::types::{HfCausalLmDecodeScratch, HfCausalLmModel};

use crate::engine::hf_cuda::run_loaded_hf_layer_on_cuda;
use crate::engine::hf_cuda_decode::run::{
    run_hf_causal_lm_cuda_prompt_decode, run_hf_causal_lm_cuda_seed_decode,
};
use crate::engine::tests::hf_fixture::{
    remove_hf_checkpoint_dir, write_cycle_hf_checkpoint_dir, write_kv_hf_checkpoint_dir,
};

#[test]
fn cuda_loaded_hf_layer_matches_cpu_exact_layer() {
    let loaded = load_hf_causal_lm_smoke_fixture().unwrap();
    let summary = run_loaded_hf_layer_on_cuda(&loaded.model, 0, TokenId(0)).unwrap();

    if summary.cuda.status != SmokeStatus::Ok {
        return;
    }

    assert!(summary.passed());
    assert_eq!(summary.layer_index, 0);
    assert_eq!(summary.hidden, loaded.model.metadata().hidden_size);
    assert_eq!(
        summary.cuda.hidden as usize,
        loaded.model.metadata().hidden_size
    );
    assert_eq!(summary.cuda.kernel_launches, 1);
    assert_eq!(summary.cuda.sync_calls, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(summary.output_hash, summary.expected_hash);
    assert_ne!(summary.output_hash, 0);
    assert!(summary.to_json().contains("\"bit_parity\":true"));
}

#[test]
fn cuda_loaded_hf_seed_decode_matches_cpu_exact_decode() {
    let loaded = load_hf_causal_lm_smoke_fixture().unwrap();
    let summary = run_hf_causal_lm_cuda_seed_decode(&loaded.model, TokenId(0), 4).unwrap();

    if summary.status != SmokeStatus::Ok {
        return;
    }

    assert!(summary.passed());
    assert_eq!(summary.tokens, summary.expected_tokens);
    assert_eq!(summary.ledger_count, 4);
    assert_eq!(summary.device_events, 4);
    assert_eq!(summary.copy_events, 2);
    assert_eq!(summary.hard_syncs, 1);
    assert_eq!(summary.execution_decisions, 4);
    assert_eq!(summary.graph_replays, 4);
    assert!(summary.graph_nodes > 0);
    assert_eq!(summary.graph_launches, 4);
    assert_eq!(summary.graph_replay_events, 4);
    assert_eq!(summary.kernel_launches, 4);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.host_causality_edges, 0);
    assert!(summary.resident_kv_bytes > 0);
    assert_eq!(summary.kv_tokens, 4);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(summary.output_hash, summary.expected_hash);
    assert!(summary.h2d_bytes >= summary.resident_weight_bytes);
    assert!(summary.d2h_bytes > 0);
    assert!(summary.to_json().contains("\"parity\":true"));
}

#[test]
fn cuda_loaded_hf_seed_decode_uses_chain_for_multi_layer_model() {
    let dir = write_cycle_hf_checkpoint_dir("nerva-hf-cuda-chain", 2);
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let summary = run_hf_causal_lm_cuda_seed_decode(&loaded.model, TokenId(0), 4).unwrap();
    remove_hf_checkpoint_dir(&dir);

    if summary.status != SmokeStatus::Ok {
        return;
    }

    assert!(summary.passed());
    assert_eq!(loaded.model.layer_count(), 2);
    assert_eq!(summary.tokens, summary.expected_tokens);
    assert_eq!(summary.ledger_count, 4);
    assert_eq!(summary.device_events, 4);
    assert_eq!(summary.copy_events, 2);
    assert_eq!(summary.hard_syncs, 1);
    assert_eq!(summary.execution_decisions, 4);
    assert_eq!(summary.graph_replays, 4);
    assert!(summary.graph_nodes > 0);
    assert_eq!(summary.graph_launches, 4);
    assert_eq!(summary.graph_replay_events, 4);
    assert_eq!(summary.kernel_launches, 4);
    assert_eq!(summary.sync_calls, 1);
    assert_eq!(summary.host_causality_edges, 0);
    assert!(summary.resident_kv_bytes > 0);
    assert_eq!(summary.kv_tokens, 4);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(summary.output_hash, summary.expected_hash);
}

#[test]
fn cuda_loaded_hf_seed_decode_matches_kv_context_model() {
    let dir = write_kv_hf_checkpoint_dir("nerva-hf-cuda-kv");
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let mut seed_scratch =
        HfCausalLmDecodeScratch::new(loaded.model.shape(), loaded.model.metadata().vocab_size)
            .unwrap();
    let (seed_only_tokens, _) = loaded
        .model
        .decode_greedy(TokenId(0), 4, &mut seed_scratch)
        .unwrap();
    let summary = run_hf_causal_lm_cuda_seed_decode(&loaded.model, TokenId(0), 4).unwrap();
    remove_hf_checkpoint_dir(&dir);

    if summary.status != SmokeStatus::Ok {
        return;
    }

    assert!(summary.passed());
    assert_eq!(summary.tokens, summary.expected_tokens);
    assert_ne!(summary.expected_tokens, seed_only_tokens);
    assert_eq!(summary.graph_replays, 4);
    assert_eq!(summary.graph_replay_events, 4);
    assert!(summary.resident_kv_bytes > 0);
    assert_eq!(summary.kv_tokens, 4);
    assert_eq!(summary.host_causality_edges, 0);
    assert_eq!(summary.hot_path_allocations, 0);
}

#[test]
fn cuda_loaded_hf_prompt_decode_uses_full_prompt_context() {
    let dir = write_kv_hf_checkpoint_dir("nerva-hf-cuda-prompt-kv");
    let loaded = HfCausalLmModel::load_from_hf_dir(&dir).unwrap();
    let prompt = [TokenId(0), TokenId(1)];
    let summary = run_hf_causal_lm_cuda_prompt_decode(&loaded.model, &prompt, 3).unwrap();
    let mut seed_scratch =
        HfCausalLmDecodeScratch::new(loaded.model.shape(), loaded.model.metadata().vocab_size)
            .unwrap();
    let (seed_only_tokens, _) = loaded
        .model
        .decode_greedy(TokenId(1), 3, &mut seed_scratch)
        .unwrap();
    remove_hf_checkpoint_dir(&dir);

    if summary.status != SmokeStatus::Ok {
        return;
    }

    assert!(summary.passed());
    assert_eq!(summary.tokens, summary.expected_tokens);
    assert_ne!(summary.expected_tokens, seed_only_tokens);
    assert_eq!(summary.graph_replays, 4);
    assert_eq!(summary.kv_tokens, 4);
    assert_eq!(summary.host_causality_edges, 0);
    assert_eq!(summary.hot_path_allocations, 0);
}
