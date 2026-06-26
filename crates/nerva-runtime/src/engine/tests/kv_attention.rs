use nerva_core::types::error::NervaError;

use crate::engine::kv_attention::config::TieredKvAttentionProbeConfig;
use crate::engine::kv_attention::summary::TieredKvAttentionProbeStatus;
use crate::engine::runtime::{Runtime, RuntimeConfig};

#[test]
fn tiered_kv_attention_probe_executes_against_resident_pages() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_tiered_kv_attention_probe(TieredKvAttentionProbeConfig::default())
        .unwrap();

    assert_eq!(summary.status, TieredKvAttentionProbeStatus::Ok);
    assert_eq!(summary.pages, 2);
    assert_eq!(summary.tokens, 4);
    assert_eq!(summary.dram_pages, 1);
    assert_eq!(summary.vram_pages, 1);
    assert!(summary.parity);
    assert_eq!(summary.max_abs_error, 0.0);
    assert_eq!(summary.execution_decisions, 2);
    assert_eq!(summary.runtime_timestamp_decisions, 2);
    assert_eq!(summary.measured_candidate_costs, 2);
    assert_eq!(summary.estimated_candidate_costs, 4);
    assert_eq!(summary.block_version_dependencies, 2);
    assert_eq!(summary.cpu_block_events, 1);
    assert_eq!(summary.device_block_events, 1);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"parity\":true"));
    assert!(
        summary
            .to_json()
            .contains("\"block_version_dependencies\":2")
    );
    assert!(summary.to_json().contains("\"measured_candidate_costs\":2"));
}

#[test]
fn tiered_kv_attention_probe_rejects_page_too_small_for_payload() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let err = runtime
        .run_tiered_kv_attention_probe(TieredKvAttentionProbeConfig {
            page_bytes: 8,
            ..TieredKvAttentionProbeConfig::default()
        })
        .unwrap_err();

    assert!(matches!(err, NervaError::InvalidArgument { .. }));
}
