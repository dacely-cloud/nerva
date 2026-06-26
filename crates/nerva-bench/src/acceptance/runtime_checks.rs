use nerva_core::types::id::token::TokenId;
use nerva_runtime::engine::hot_path::status::HotPathGuardStatus;
use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::engine::synthetic::config::SyntheticDecodeConfig;
use nerva_runtime::engine::synthetic::summary::SyntheticDecodeStatus;
use nerva_runtime::residency::budget::ResidencyBudget;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_static_arenas(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.static_arena_probe(ResidencyBudget::new(1024, 2048, 4096)) {
        Ok(summary) => report.push(
            "static_arenas",
            summary.device_capacity_bytes == 1024
                && summary.pinned_host_capacity_bytes == 2048
                && summary.host_capacity_bytes == 4096
                && summary.bootstrap_blocks == 3
                && summary.ready_blocks == 3
                && summary.hot_path_rejections == 3
                && summary.hot_path_allocation_attempts == 3
                && summary.usage_preserved_after_rejections,
            format!(
                "device_capacity={} pinned_host_capacity={} host_capacity={} device_used={} pinned_host_used={} host_used={} bootstrap_blocks={} ready_blocks={} hot_path_rejections={} hot_path_allocation_attempts={} usage_preserved={}",
                summary.device_capacity_bytes,
                summary.pinned_host_capacity_bytes,
                summary.host_capacity_bytes,
                summary.device_used_bytes,
                summary.pinned_host_used_bytes,
                summary.host_used_bytes,
                summary.bootstrap_blocks,
                summary.ready_blocks,
                summary.hot_path_rejections,
                summary.hot_path_allocation_attempts,
                summary.usage_preserved_after_rejections,
            ),
        ),
        Err(err) => report.push("static_arenas", false, format!("{err:?}")),
    }
}

pub(crate) fn push_hot_path_guard(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_hot_path_guard_probe(ResidencyBudget::new(1024, 2048, 4096)) {
        Ok(summary) => report.push(
            "hot_path_guard",
            summary.status == HotPathGuardStatus::Ok
                && summary.passed()
                && summary.entered_scopes == 2
                && summary.exited_scopes == 2
                && summary.active_scopes_after_probe == 0
                && summary.clean_scope_allocation_attempts == 0
                && summary.deliberate_allocation_attempts == 3
                && summary.deliberate_rejections == 3
                && summary.ledger_allocation_events == 3
                && summary.ledger_hot_path_allocations == 3
                && summary.release_to_system_calls == 0
                && summary.usage_preserved_after_rejections,
            format!(
                "status={:?} entered_scopes={} exited_scopes={} active_scopes={} clean_scope_allocation_attempts={} deliberate_attempts={} deliberate_rejections={} ledger_allocation_events={} ledger_hot_path_allocations={} attempted_bytes={} release_to_system_calls={} usage_preserved={}",
                summary.status,
                summary.entered_scopes,
                summary.exited_scopes,
                summary.active_scopes_after_probe,
                summary.clean_scope_allocation_attempts,
                summary.deliberate_allocation_attempts,
                summary.deliberate_rejections,
                summary.ledger_allocation_events,
                summary.ledger_hot_path_allocations,
                summary.attempted_bytes,
                summary.release_to_system_calls,
                summary.usage_preserved_after_rejections,
            ),
        ),
        Err(err) => report.push("hot_path_guard", false, format!("{err:?}")),
    }
}

pub(crate) fn push_synthetic_decode(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_synthetic_decode(SyntheticDecodeConfig::new(1024, 64, TokenId(1))) {
        Ok(summary) => {
            let transaction_passed = matches!(summary.status, SyntheticDecodeStatus::Ok)
                && summary.steps == 1024
                && summary.graph_replays == summary.steps
                && summary.graph_replay_events == summary.steps
                && summary.kernel_events >= summary.steps
                && summary.device_events == summary.steps
                && summary.copy_events == summary.steps
                && summary.host_wait_events == summary.steps
                && summary.graph_replay_latency_ns > 0
                && summary.device_latency_ns > 0
                && summary.copy_latency_ns > 0
                && summary.host_wait_latency_ns > 0
                && summary.hot_path_allocations == 0;
            report.push(
                "synthetic_transaction",
                transaction_passed,
                format!(
                    "steps={} graph_replays={} graph_events={} kernel_events={} device_events={} copy_events={} host_wait_events={} graph_ns={} device_ns={} copy_ns={} host_wait_ns={} hot_path_allocations={}",
                    summary.steps,
                    summary.graph_replays,
                    summary.graph_replay_events,
                    summary.kernel_events,
                    summary.device_events,
                    summary.copy_events,
                    summary.host_wait_events,
                    summary.graph_replay_latency_ns,
                    summary.device_latency_ns,
                    summary.copy_latency_ns,
                    summary.host_wait_latency_ns,
                    summary.hot_path_allocations,
                ),
            );

            let passed = matches!(summary.status, SyntheticDecodeStatus::Ok)
                && summary.steps == 1024
                && summary.graph_replays == 1024
                && summary.observed_tokens == 1024
                && summary.observed_token_hash != 0
                && summary.token_ring_slots_touched == 64
                && summary.token_ring_reuses == 960
                && summary.token_ring_max_slot_version == 16
                && summary.soft_visibility_syncs == 1024
                && summary.device_timeline_active_ns > 0
                && summary.device_timeline_idle_ns == 0
                && summary.hot_path_allocations == 0
                && summary.stale_tokens == 0
                && summary.missing_tokens == 0
                && summary.extra_tokens == 0
                && summary.mismatched_tokens == 0
                && summary.host_causality_edges == 0;
            report.push(
                "synthetic_device_token",
                passed,
                format!(
                    "steps={} observed={} observed_token_hash={} ring_slots={} ring_reuses={} ring_max_version={} soft_visibility_syncs={} hot_path_allocations={} stale={} missing={} extra={} mismatched={} host_causality_edges={} gpu_idle_ns={}",
                    summary.steps,
                    summary.observed_tokens,
                    summary.observed_token_hash,
                    summary.token_ring_slots_touched,
                    summary.token_ring_reuses,
                    summary.token_ring_max_slot_version,
                    summary.soft_visibility_syncs,
                    summary.hot_path_allocations,
                    summary.stale_tokens,
                    summary.missing_tokens,
                    summary.extra_tokens,
                    summary.mismatched_tokens,
                    summary.host_causality_edges,
                    summary.device_timeline_idle_ns,
                ),
            );
        }
        Err(err) => {
            let details = format!("{err:?}");
            report.push("synthetic_transaction", false, details.clone());
            report.push("synthetic_device_token", false, details);
        }
    }
}

pub(crate) fn push_critical_path(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_critical_path_probe() {
        Ok(summary) => report.push(
            "critical_path_observability",
            summary.proves_host_wait_not_gpu_idle()
                && summary.host_event_wait_ns == 1
                && summary.gpu_idle_ns == 0
                && summary.device_timeline_active_ns == 3
                && summary.host_wait_events == 1
                && summary.device_timeline_spans == 1
                && summary.host_wait_gpu_idle_sources_separate
                && !summary.estimated_presented_as_measured,
            format!(
                "token_index={} wall_ns={} host_event_wait_ns={} gpu_idle_ns={} device_active_ns={} host_wait_events={} device_timeline_spans={} estimated_events={} runtime_timestamp_events={} gpu_event_events={} estimated_presented_as_measured={} proves_host_wait_not_gpu_idle={}",
                summary.token_index,
                summary.wall_latency_ns,
                summary.host_event_wait_ns,
                summary.gpu_idle_ns,
                summary.device_timeline_active_ns,
                summary.host_wait_events,
                summary.device_timeline_spans,
                summary.estimated_event_count,
                summary.runtime_timestamp_event_count,
                summary.gpu_event_count,
                summary.estimated_presented_as_measured,
                summary.proves_host_wait_not_gpu_idle(),
            ),
        ),
        Err(err) => report.push("critical_path_observability", false, format!("{err:?}")),
    }
}
