use nerva_model::tiny::output::TinyGreedyDecodeSummary;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_tiny_decode_check(
    report: &mut AcceptanceReport,
    summary: &TinyGreedyDecodeSummary,
) {
    let cuda_decode = nerva_cuda::decode::probe::tiny_decode_smoke(summary.steps as u32, 4, 0);
    let cuda_decode_passed = format!("{:?}", cuda_decode.status) == "Ok"
        && cuda_decode.steps == summary.steps as u32
        && cuda_decode.ring_capacity == 4
        && cuda_decode.seed_token == summary.seed_token.0
        && cuda_decode.vocab_size == summary.vocab_size as u32
        && cuda_decode.hidden == 2
        && cuda_decode.last_token == summary.tokens.last().map(|token| token.0)
        && cuda_decode.graph_replays == summary.steps as u64
        && cuda_decode.graph_launches == summary.steps as u64
        && cuda_decode.kernel_launches == summary.steps as u64
        && cuda_decode.token_ledgers == summary.steps as u64
        && cuda_decode.graph_replay_events == summary.steps as u64
        && cuda_decode.device_activity_events == summary.steps as u64
        && cuda_decode.copy_events == summary.steps as u64
        && cuda_decode.soft_visibility_syncs == summary.steps as u64
        && cuda_decode.hard_syncs == 0
        && cuda_decode.host_event_wait_ns > 0
        && cuda_decode.gpu_active_ns > 0
        && cuda_decode.gpu_idle_ns == 0
        && cuda_decode.wall_latency_ns > 0
        && cuda_decode.host_event_wait_ns != cuda_decode.gpu_idle_ns
        && cuda_decode.observed_tokens == summary.steps as u64
        && cuda_decode.observed_token_hash == summary.output_hash
        && cuda_decode.token_ring_slots_touched == 4
        && cuda_decode.token_ring_reuses == 4
        && cuda_decode.token_ring_max_slot_version == 2
        && cuda_decode.resident_weight_bytes == 64
        && cuda_decode.h2d_bytes >= cuda_decode.resident_weight_bytes
        && cuda_decode.d2h_bytes > 0
        && cuda_decode.sync_calls == summary.steps as u64
        && cuda_decode.hot_path_allocations == 0
        && cuda_decode.stale_tokens == 0
        && cuda_decode.missing_tokens == 0
        && cuda_decode.extra_tokens == 0
        && cuda_decode.mismatched_tokens == 0
        && cuda_decode.host_causality_edges == 0;
    report.push(
        "cuda_tiny_decode_model",
        cuda_decode_passed,
        format!(
            "status={:?} steps={} ring_capacity={} graph_replays={} graph_nodes={} token_ledgers={} graph_replay_events={} device_activity_events={} copy_events={} soft_visibility_syncs={} hard_syncs={} host_event_wait_ns={} gpu_active_ns={} gpu_idle_ns={} wall_latency_ns={} observed={} observed_token_hash={} reference_hash={} last_token={:?} ring_slots={} ring_reuses={} ring_max_version={} resident_weight_bytes={} H2D_bytes={} D2H_bytes={} kernel_launches={} sync_calls={} hot_path_allocations={} stale={} missing={} extra={} mismatched={} host_causality_edges={} error={}",
            cuda_decode.status,
            cuda_decode.steps,
            cuda_decode.ring_capacity,
            cuda_decode.graph_replays,
            cuda_decode.graph_nodes,
            cuda_decode.token_ledgers,
            cuda_decode.graph_replay_events,
            cuda_decode.device_activity_events,
            cuda_decode.copy_events,
            cuda_decode.soft_visibility_syncs,
            cuda_decode.hard_syncs,
            cuda_decode.host_event_wait_ns,
            cuda_decode.gpu_active_ns,
            cuda_decode.gpu_idle_ns,
            cuda_decode.wall_latency_ns,
            cuda_decode.observed_tokens,
            cuda_decode.observed_token_hash,
            summary.output_hash,
            cuda_decode.last_token,
            cuda_decode.token_ring_slots_touched,
            cuda_decode.token_ring_reuses,
            cuda_decode.token_ring_max_slot_version,
            cuda_decode.resident_weight_bytes,
            cuda_decode.h2d_bytes,
            cuda_decode.d2h_bytes,
            cuda_decode.kernel_launches,
            cuda_decode.sync_calls,
            cuda_decode.hot_path_allocations,
            cuda_decode.stale_tokens,
            cuda_decode.missing_tokens,
            cuda_decode.extra_tokens,
            cuda_decode.mismatched_tokens,
            cuda_decode.host_causality_edges,
            cuda_decode.error.as_deref().unwrap_or("none"),
        ),
    );
}

pub(crate) fn push_prerequisite_failure(report: &mut AcceptanceReport, details: &str) {
    report.push(
        "cuda_tiny_decode_model",
        false,
        format!("tiny reference model prerequisite failed: {details}"),
    );
}
