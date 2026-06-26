use nerva_core::types::memory::tier::MemoryTier;

use crate::engine::residency::ResidencyBudget;
use crate::engine::runtime::{Runtime, RuntimeConfig};

#[test]
fn runtime_creates_residency_registry_from_budget() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let registry = runtime.block_registry(ResidencyBudget::new(1024, 2048, 4096));
    assert_eq!(registry.remaining_bytes(MemoryTier::Vram), Some(1024));
    assert_eq!(registry.remaining_bytes(MemoryTier::PinnedDram), Some(2048));
    assert_eq!(registry.remaining_bytes(MemoryTier::Dram), Some(4096));
}

#[test]
fn runtime_creates_static_arenas_from_budget() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let arenas = runtime.static_arenas(ResidencyBudget::new(1024, 2048, 4096));
    assert_eq!(arenas.device().capacity(), 1024);
    assert_eq!(arenas.pinned_host().capacity(), 2048);
    assert_eq!(arenas.host().capacity(), 4096);
}

#[test]
fn static_arena_probe_bootstraps_and_rejects_hot_path_allocations() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .static_arena_probe(ResidencyBudget::new(1024, 2048, 4096))
        .unwrap();

    assert_eq!(summary.bootstrap_blocks, 3);
    assert_eq!(summary.ready_blocks, 3);
    assert_eq!(summary.device_used_bytes, 256);
    assert_eq!(summary.pinned_host_used_bytes, 256);
    assert_eq!(summary.host_used_bytes, 512);
    assert_eq!(summary.hot_path_rejections, 3);
    assert_eq!(summary.hot_path_allocation_attempts, 3);
    assert!(summary.usage_preserved_after_rejections);
    assert!(summary.to_json().contains("\"ready_blocks\":3"));
}
