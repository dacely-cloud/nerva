use nerva_core::types::error::NervaError;

use crate::engine::kv_probe::{KvResidencyProbeConfig, KvResidencyProbeStatus};
use crate::engine::runtime::{Runtime, RuntimeConfig};

#[test]
fn kv_residency_probe_exercises_prefetch_demote_and_evict_paths() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_kv_residency_probe(KvResidencyProbeConfig::default())
        .unwrap();

    assert_eq!(summary.status, KvResidencyProbeStatus::Ok);
    assert_eq!(summary.decisions, 4);
    assert_eq!(summary.prefetches, 2);
    assert_eq!(summary.demotions, 1);
    assert_eq!(summary.evictions, 1);
    assert_eq!(summary.copy_events, 3);
    assert_eq!(summary.prefetch_events, 2);
    assert_eq!(summary.eviction_events, 2);
    assert_eq!(summary.stall_events, 3);
    assert_eq!(summary.copy_bytes, 384);
    assert_eq!(summary.changed_bytes, 384);
    assert_eq!(summary.visible_stall_ns, 684);
    assert_eq!(summary.total_latency_ns, 684);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(summary.vram_used_bytes, 256);
    assert_eq!(summary.dram_used_bytes, 256);
    assert!(summary.to_json().contains("\"prefetches\":2"));
    assert!(summary.to_json().contains("\"stall_events\":3"));
}

#[test]
fn kv_residency_probe_rejects_too_few_pages() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let err = runtime
        .run_kv_residency_probe(KvResidencyProbeConfig {
            pages: 3,
            ..KvResidencyProbeConfig::default()
        })
        .unwrap_err();
    assert!(matches!(err, NervaError::InvalidArgument { .. }));
}
