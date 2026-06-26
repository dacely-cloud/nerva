use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::transport::contract::summary::TransportContractStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_transport_contract(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_transport_contract_probe() {
        Ok(summary) => report.push(
            "transport_backend_contract",
            matches!(summary.status, TransportContractStatus::Ok)
                && summary.passed()
                && summary.registrations == 2
                && summary.registered_entries == 2
                && summary.preposted_receives == 0
                && summary.sends == 1
                && summary.completions == 1
                && summary.bytes_completed == 32 * 1024
                && summary.unposted_send_rejections == 1
                && summary.stale_version_rejections == 1
                && summary.descriptor_rejections == 1
                && summary.pre_visibility_consume_rejections == 1
                && summary.visibility_fences == 1
                && summary.visible_consumes == 1
                && summary.per_transfer_registrations == 0
                && summary.transport_events == 1
                && summary.phase_handoff_syncs == 1
                && summary.hot_path_allocations == 0,
            format!(
                "backend={} registrations={} registered_entries={} preposted_receives={} sends={} completions={} bytes_completed={} unposted_send_rejections={} stale_version_rejections={} descriptor_rejections={} pre_visibility_consume_rejections={} visibility_fences={} visible_consumes={} per_transfer_registrations={} transport_events={} phase_handoff_syncs={} hot_path_allocations={}",
                summary.backend,
                summary.registrations,
                summary.registered_entries,
                summary.preposted_receives,
                summary.sends,
                summary.completions,
                summary.bytes_completed,
                summary.unposted_send_rejections,
                summary.stale_version_rejections,
                summary.descriptor_rejections,
                summary.pre_visibility_consume_rejections,
                summary.visibility_fences,
                summary.visible_consumes,
                summary.per_transfer_registrations,
                summary.transport_events,
                summary.phase_handoff_syncs,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push(
            "transport_backend_contract",
            false,
            format!("transport contract probe failed: {err:?}"),
        ),
    }
}
