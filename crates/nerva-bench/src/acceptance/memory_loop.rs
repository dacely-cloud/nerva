use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::memory_loop::summary::MemoryLoopStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_memory_fabric_loop(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_memory_loop_probe() {
        Ok(summary) => report.push(
            "memory_fabric_loop",
            matches!(summary.status, MemoryLoopStatus::Ok)
                && summary.passed()
                && summary.queue_overflows == 0
                && summary.disk_read_tasks > 0
                && summary.prefetch_tasks > 0
                && summary.staging_tasks > 0
                && summary.eviction_tasks > 0
                && summary.transport_prepare_tasks > 0
                && summary.residency_decisions == summary.tasks
                && summary.pageable_copies == 0
                && summary.per_token_registrations == 0
                && summary.page_faults == 0
                && summary.hot_path_allocations == 0,
            format!(
                "tasks={} queue_capacity={} max_inflight={} disk_reads={} prefetches={} stagings={} evictions={} transport_prepares={} ready_blocks={} prefetch_events={} eviction_events={} copy_events={} transport_events={} phase_handoff_syncs={} residency_decisions={} actual_visible_ns={} overlapped_ns={} pageable_copies={} per_token_registrations={} page_faults={} hot_path_allocations={}",
                summary.tasks,
                summary.queue_capacity,
                summary.max_inflight,
                summary.disk_read_tasks,
                summary.prefetch_tasks,
                summary.staging_tasks,
                summary.eviction_tasks,
                summary.transport_prepare_tasks,
                summary.ready_blocks,
                summary.prefetch_events,
                summary.eviction_events,
                summary.copy_events,
                summary.transport_events,
                summary.phase_handoff_syncs,
                summary.residency_decisions,
                summary.actual_visible_ns,
                summary.overlapped_ns,
                summary.pageable_copies,
                summary.per_token_registrations,
                summary.page_faults,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("memory_fabric_loop", false, format!("{err:?}")),
    }
}
