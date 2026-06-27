use crate::causal_lm::smoke::hf_causal_lm_safetensors_smoke;
use crate::causal_lm::summary::HfCausalLmSmokeStatus;

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
