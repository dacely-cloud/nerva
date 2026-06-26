use nerva_core::types::id::TokenId;
use nerva_runtime::engine::residency::ResidencyBudget;
use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::engine::synthetic::config::SyntheticDecodeConfig;
use nerva_runtime::engine::synthetic::summary::SyntheticDecodeStatus;

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
