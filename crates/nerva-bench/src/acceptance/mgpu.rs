use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::mgpu::config::MultiGpuNodeConfig;
use nerva_runtime::mgpu::summary::MultiGpuNodeStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_multi_gpu_node(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_multi_gpu_node_probe(MultiGpuNodeConfig::reference_2080ti_stage()) {
        Ok(summary) => report.push(
            "same_node_multi_gpu_islands",
            matches!(summary.status, MultiGpuNodeStatus::Ok)
                && summary.passed()
                && summary.gpu_islands == summary.gpu_count
                && !summary.aggregate_vram_pool_claimed
                && summary.coherent_vram_allocation_claims == 0
                && summary.max_single_allocation_bytes <= summary.local_vram_bytes_per_gpu
                && summary.activation_only_boundaries == summary.local_boundaries
                && summary.inter_gpu_weight_bytes == 0
                && summary.all_reduce_bytes == 0
                && summary.phase_handoff_syncs == u64::from(summary.local_boundaries)
                && summary.hot_path_allocations == 0,
            format!(
                "gpus={} islands={} local_vram_bytes_per_gpu={} aggregate_vram_bytes={} aggregate_vram_pool_claimed={} coherent_vram_allocation_claims={} max_single_allocation_bytes={} stage_weight_bytes={} hot_weight_cache_bytes={} dram_weight_backing_bytes={} kv_owner_count={} boundaries={} activation_bytes_per_boundary={} activation_bytes_moved={} inter_gpu_weight_bytes={} all_reduce_bytes={} egress_gpu={} nic_near_egress={} execution_decisions={} phase_handoff_syncs={} hot_path_allocations={}",
                summary.gpu_count,
                summary.gpu_islands,
                summary.local_vram_bytes_per_gpu,
                summary.aggregate_vram_bytes,
                summary.aggregate_vram_pool_claimed,
                summary.coherent_vram_allocation_claims,
                summary.max_single_allocation_bytes,
                summary.stage_weight_bytes,
                summary.hot_weight_cache_bytes,
                summary.dram_weight_backing_bytes,
                summary.kv_owner_count,
                summary.local_boundaries,
                summary.activation_bytes_per_boundary,
                summary.activation_bytes_moved,
                summary.inter_gpu_weight_bytes,
                summary.all_reduce_bytes,
                summary.egress_gpu,
                summary.nic_near_egress,
                summary.execution_decisions,
                summary.phase_handoff_syncs,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("same_node_multi_gpu_islands", false, format!("{err:?}")),
    }
}
