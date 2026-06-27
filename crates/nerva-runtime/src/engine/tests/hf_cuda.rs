use nerva_core::types::id::token::TokenId;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::causal_lm::smoke::load_hf_causal_lm_smoke_fixture;

use crate::engine::hf_cuda::run_loaded_hf_layer_on_cuda;
use crate::engine::hf_cuda_decode::run::run_hf_causal_lm_cuda_seed_decode;

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
    assert_eq!(summary.device_events, 8);
    assert_eq!(summary.copy_events, 16);
    assert_eq!(summary.hard_syncs, 8);
    assert_eq!(summary.execution_decisions, 8);
    assert_eq!(summary.kernel_launches, 8);
    assert_eq!(summary.sync_calls, 8);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(summary.output_hash, summary.expected_hash);
    assert!(summary.h2d_bytes >= summary.resident_weight_bytes);
    assert!(summary.d2h_bytes > 0);
    assert!(summary.to_json().contains("\"parity\":true"));
}
