use nerva_runtime::engine::runtime::Runtime;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_shared_queue(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_shared_work_queue_probe() {
        Ok(summary) => report.push(
            "shared_work_queue",
            summary.passed()
                && summary.queue_capacity == 4
                && summary.queue_blocks_ready == 2
                && summary.atomic_control_blocks == 2
                && summary.descriptors_posted == 4
                && summary.descriptors_completed == 4
                && summary.completion_records == 4
                && summary.queue_full_rejections == 1
                && summary.wrong_producer_rejections == 1
                && summary.wrong_consumer_rejections == 1
                && summary.bulk_payload_rejections == 1
                && summary.payload_bytes_in_queue == 0
                && summary.phase_handoff_syncs == summary.descriptors_completed
                && summary.hot_path_allocations == 0,
            format!(
                "queue_capacity={} queue_blocks_ready={} atomic_control_blocks={} descriptors_posted={} descriptors_completed={} completion_records={} queue_full_rejections={} wrong_producer_rejections={} wrong_consumer_rejections={} bulk_payload_rejections={} payload_bytes_in_queue={} referenced_block_bytes={} phase_handoff_syncs={} hot_path_allocations={}",
                summary.queue_capacity,
                summary.queue_blocks_ready,
                summary.atomic_control_blocks,
                summary.descriptors_posted,
                summary.descriptors_completed,
                summary.completion_records,
                summary.queue_full_rejections,
                summary.wrong_producer_rejections,
                summary.wrong_consumer_rejections,
                summary.bulk_payload_rejections,
                summary.payload_bytes_in_queue,
                summary.referenced_block_bytes,
                summary.phase_handoff_syncs,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("shared_work_queue", false, format!("{err:?}")),
    }
}
