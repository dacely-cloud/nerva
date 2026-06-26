use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::mgpu::config::MultiGpuNodeConfig;
use crate::mgpu::plan::plan_multi_gpu_node;

#[test]
fn multi_gpu_plan_keeps_gpu_memory_islands_separate() {
    let config = MultiGpuNodeConfig::reference_2080ti_stage();
    let plan = plan_multi_gpu_node(config).unwrap();

    assert_eq!(plan.islands.len(), config.gpu_count as usize);
    assert_eq!(
        plan.boundaries.len(),
        config.gpu_count.saturating_sub(1) as usize
    );
    assert!(!plan.aggregate_vram_pool_claimed);
    assert_eq!(plan.coherent_vram_allocation_claims, 0);
    for island in &plan.islands {
        assert!(island.max_single_allocation_bytes <= island.local_vram_bytes);
        assert!(island.hot_weight_bytes <= island.local_vram_bytes);
        assert!(island.kv_bytes <= island.local_vram_bytes);
    }
    for boundary in &plan.boundaries {
        assert_eq!(boundary.moved_weight_bytes, 0);
        assert_eq!(boundary.all_reduce_bytes, 0);
        assert!(boundary.phase_handoff_required);
    }
}

#[test]
fn multi_gpu_probe_reports_activation_only_boundaries() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_multi_gpu_node_probe(MultiGpuNodeConfig::reference_2080ti_stage())
        .unwrap();

    assert!(summary.passed());
    assert_eq!(summary.gpu_count, 8);
    assert_eq!(summary.gpu_islands, 8);
    assert_eq!(summary.local_boundaries, 7);
    assert_eq!(summary.activation_only_boundaries, summary.local_boundaries);
    assert_eq!(summary.inter_gpu_weight_bytes, 0);
    assert_eq!(summary.all_reduce_bytes, 0);
    assert!(!summary.aggregate_vram_pool_claimed);
    assert_eq!(summary.coherent_vram_allocation_claims, 0);
    assert_eq!(summary.execution_decisions, 8);
    assert_eq!(summary.device_events, 8);
    assert_eq!(summary.copy_events, 7);
    assert_eq!(summary.phase_handoff_syncs, 7);
    assert_eq!(summary.hot_path_allocations, 0);
}

#[test]
fn multi_gpu_plan_rejects_aggregate_hot_cache_over_local_vram() {
    let mut config = MultiGpuNodeConfig::reference_2080ti_stage();
    config.hot_weight_cache_bytes_per_gpu = config.local_vram_bytes_per_gpu + 1;

    let err = plan_multi_gpu_node(config).unwrap_err();
    assert!(format!("{err:?}").contains("hot cache cannot exceed local GPU VRAM"));
}
