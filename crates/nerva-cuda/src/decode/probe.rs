use crate::decode::ffi::{CUDA_ERROR_NO_DEVICE, NervaCudaTinyDecodeResult, run_tiny_decode_smoke};
use crate::decode::summary::CudaTinyDecodeSummary;
use crate::smoke::status::SmokeStatus;

pub fn tiny_decode_smoke(steps: u32, ring_capacity: u32, seed_token: u32) -> CudaTinyDecodeSummary {
    let mut out = NervaCudaTinyDecodeResult::default();
    let return_code = run_tiny_decode_smoke(steps, ring_capacity, seed_token, &mut out);

    if return_code == 0
        && out.status == 0
        && out.steps == steps
        && out.ring_capacity == ring_capacity
        && out.seed_token == seed_token
        && out.vocab_size == 4
        && out.hidden == 2
        && out.graph_replays == steps as u64
        && out.graph_launches == steps as u64
        && out.kernel_launches == steps as u64
        && out.token_ledgers == steps as u64
        && out.graph_replay_events == steps as u64
        && out.device_activity_events == steps as u64
        && out.copy_events == steps as u64
        && out.soft_visibility_syncs == steps as u64
        && out.hard_syncs == 0
        && out.host_event_wait_ns > 0
        && out.gpu_active_ns > 0
        && out.gpu_idle_ns == 0
        && out.wall_latency_ns > 0
        && out.observed_tokens == steps as u64
        && out.observed_token_hash != 0
        && out.hot_path_allocations == 0
        && out.stale_tokens == 0
        && out.missing_tokens == 0
        && out.extra_tokens == 0
        && out.mismatched_tokens == 0
        && out.host_causality_edges == 0
    {
        return CudaTinyDecodeSummary {
            status: SmokeStatus::Ok,
            steps: out.steps,
            ring_capacity: out.ring_capacity,
            seed_token: out.seed_token,
            vocab_size: out.vocab_size,
            hidden: out.hidden,
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
            resident_weight_bytes: out.resident_weight_bytes,
            device_arena_bytes: out.device_arena_bytes,
            pinned_host_bytes: out.pinned_host_bytes,
            h2d_bytes: out.h2d_bytes,
            d2h_bytes: out.d2h_bytes,
            graph_launches: out.graph_launches,
            sync_calls: out.sync_calls,
            kernel_launches: out.kernel_launches,
            hot_path_allocations: out.hot_path_allocations,
            token_ledgers: out.token_ledgers,
            graph_replay_events: out.graph_replay_events,
            device_activity_events: out.device_activity_events,
            copy_events: out.copy_events,
            soft_visibility_syncs: out.soft_visibility_syncs,
            hard_syncs: out.hard_syncs,
            host_event_wait_ns: out.host_event_wait_ns,
            gpu_active_ns: out.gpu_active_ns,
            gpu_idle_ns: out.gpu_idle_ns,
            wall_latency_ns: out.wall_latency_ns,
            error: None,
        };
    }

    let reason = format!(
        "CUDA tiny decode smoke failed: return_code={} status={} cuda_error={} device_count={} steps={} ring_capacity={} seed_token={} observed={} hash={} mismatched={} graph_replays={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.steps,
        out.ring_capacity,
        out.seed_token,
        out.observed_tokens,
        out.observed_token_hash,
        out.mismatched_tokens,
        out.graph_replays,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaTinyDecodeSummary::unavailable(steps, ring_capacity, seed_token, reason)
    } else {
        CudaTinyDecodeSummary::failed(steps, ring_capacity, seed_token, reason)
    }
}
