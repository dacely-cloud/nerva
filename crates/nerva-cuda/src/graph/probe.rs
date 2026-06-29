use crate::graph::ffi::{
    CUDA_ERROR_NO_DEVICE, NervaCudaSyntheticGraphResult, run_synthetic_graph_smoke,
};
use crate::graph::summary::CudaSyntheticGraphSummary;
use crate::smoke::status::SmokeStatus;

pub fn synthetic_graph_smoke(
    steps: u32,
    ring_capacity: u32,
    seed_token: u32,
) -> CudaSyntheticGraphSummary {
    let mut out = NervaCudaSyntheticGraphResult::default();
    let return_code = run_synthetic_graph_smoke(steps, ring_capacity, seed_token, &mut out);

    if return_code == 0
        && out.status == 0
        && out.graph_replays == steps as u64
        && out.observed_tokens == steps as u64
        && out.hot_path_allocations == 0
        && out.stale_tokens == 0
        && out.missing_tokens == 0
        && out.extra_tokens == 0
        && out.mismatched_tokens == 0
        && out.host_causality_edges == 0
    {
        return CudaSyntheticGraphSummary {
            status: SmokeStatus::Ok,
            steps: out.steps,
            ring_capacity: out.ring_capacity,
            seed_token: out.seed_token,
            last_token: Some(out.last_token),
            graph_replays: out.graph_replays,
            graph_nodes: out.graph_nodes,
            observed_tokens: out.observed_tokens,
            observed_token_hash: out.observed_token_hash,
            token_ring_slots_touched: out.token_ring_slots_touched,
            token_ring_reuses: out.token_ring_reuses,
            token_ring_max_slot_version: out.token_ring_max_slot_version,
            stale_tokens: out.stale_tokens,
            missing_tokens: out.missing_tokens,
            extra_tokens: out.extra_tokens,
            mismatched_tokens: out.mismatched_tokens,
            host_causality_edges: out.host_causality_edges,
            device_arena_bytes: out.device_arena_bytes,
            pinned_host_bytes: out.pinned_host_bytes,
            graph_launches: out.graph_launches,
            sync_calls: out.sync_calls,
            d2h_bytes: out.d2h_bytes,
            hot_path_allocations: out.hot_path_allocations,
            error: None,
        };
    }

    let reason = format!(
        "CUDA synthetic graph smoke failed: return_code={} status={} cuda_error={} steps={} ring_capacity={} graph_replays={} observed={} stale={} missing={} extra={} mismatched={} host_causality_edges={}",
        return_code,
        out.status,
        out.cuda_error,
        out.steps,
        out.ring_capacity,
        out.graph_replays,
        out.observed_tokens,
        out.stale_tokens,
        out.missing_tokens,
        out.extra_tokens,
        out.mismatched_tokens,
        out.host_causality_edges,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE {
        CudaSyntheticGraphSummary::unavailable(steps, ring_capacity, seed_token, reason)
    } else {
        CudaSyntheticGraphSummary::failed(steps, ring_capacity, seed_token, reason)
    }
}
