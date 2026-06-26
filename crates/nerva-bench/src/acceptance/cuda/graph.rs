use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_transaction(report: &mut AcceptanceReport) {
    let cuda_graph = nerva_runtime::engine::cuda::cuda_synthetic_graph_smoke(1024, 64, 1);
    let cuda_graph_passed = format!("{:?}", cuda_graph.status) == "Ok"
        && cuda_graph.steps == 1024
        && cuda_graph.ring_capacity == 64
        && cuda_graph.last_token == Some(1025)
        && cuda_graph.graph_replays == 1024
        && cuda_graph.graph_launches == 1024
        && cuda_graph.graph_nodes >= 2
        && cuda_graph.observed_tokens == 1024
        && cuda_graph.observed_token_hash != 0
        && cuda_graph.token_ring_slots_touched == 64
        && cuda_graph.token_ring_reuses == 960
        && cuda_graph.token_ring_max_slot_version == 16
        && cuda_graph.sync_calls == 1024
        && cuda_graph.d2h_bytes > 0
        && cuda_graph.device_arena_bytes > 0
        && cuda_graph.pinned_host_bytes > 0
        && cuda_graph.hot_path_allocations == 0
        && cuda_graph.stale_tokens == 0
        && cuda_graph.missing_tokens == 0
        && cuda_graph.extra_tokens == 0
        && cuda_graph.mismatched_tokens == 0
        && cuda_graph.host_causality_edges == 0;
    report.push(
        "cuda_graph_transaction",
        cuda_graph_passed,
        format!(
            "status={:?} steps={} ring_capacity={} graph_replays={} graph_launches={} graph_nodes={} observed={} observed_token_hash={} ring_slots={} ring_reuses={} ring_max_version={} sync_calls={} D2H_bytes={} device_arena_bytes={} pinned_host_bytes={} hot_path_allocations={} stale={} missing={} extra={} mismatched={} host_causality_edges={} error={}",
            cuda_graph.status,
            cuda_graph.steps,
            cuda_graph.ring_capacity,
            cuda_graph.graph_replays,
            cuda_graph.graph_launches,
            cuda_graph.graph_nodes,
            cuda_graph.observed_tokens,
            cuda_graph.observed_token_hash,
            cuda_graph.token_ring_slots_touched,
            cuda_graph.token_ring_reuses,
            cuda_graph.token_ring_max_slot_version,
            cuda_graph.sync_calls,
            cuda_graph.d2h_bytes,
            cuda_graph.device_arena_bytes,
            cuda_graph.pinned_host_bytes,
            cuda_graph.hot_path_allocations,
            cuda_graph.stale_tokens,
            cuda_graph.missing_tokens,
            cuda_graph.extra_tokens,
            cuda_graph.mismatched_tokens,
            cuda_graph.host_causality_edges,
            cuda_graph.error.as_deref().unwrap_or("none"),
        ),
    );
}
