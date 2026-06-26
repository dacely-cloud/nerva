//! CUDA graph-backed tiny greedy decode smoke.

use std::os::raw::c_int;

use crate::smoke::{SmokeStatus, escape_json};

const CUDA_ERROR_NO_DEVICE: i32 = 100;

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct NervaCudaTinyDecodeResult {
    status: i32,
    cuda_error: i32,
    device_count: i32,
    steps: u32,
    ring_capacity: u32,
    seed_token: u32,
    vocab_size: u32,
    hidden: u32,
    last_token: u32,
    graph_replays: u64,
    graph_nodes: u64,
    observed_tokens: u64,
    observed_token_hash: u64,
    token_ring_slots_touched: u64,
    token_ring_reuses: u64,
    token_ring_max_slot_version: u64,
    stale_tokens: u64,
    missing_tokens: u64,
    extra_tokens: u64,
    mismatched_tokens: u64,
    host_causality_edges: u64,
    resident_weight_bytes: u64,
    device_arena_bytes: u64,
    pinned_host_bytes: u64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    graph_launches: u64,
    sync_calls: u64,
    kernel_launches: u64,
    hot_path_allocations: u64,
}

unsafe extern "C" {
    fn nerva_cuda_tiny_decode_smoke(
        steps: u32,
        ring_capacity: u32,
        seed_token: u32,
        out: *mut NervaCudaTinyDecodeResult,
    ) -> c_int;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaTinyDecodeSummary {
    pub status: SmokeStatus,
    pub steps: u32,
    pub ring_capacity: u32,
    pub seed_token: u32,
    pub vocab_size: u32,
    pub hidden: u32,
    pub last_token: Option<u32>,
    pub graph_replays: u64,
    pub graph_nodes: u64,
    pub observed_tokens: u64,
    pub observed_token_hash: u64,
    pub token_ring_slots_touched: u64,
    pub token_ring_reuses: u64,
    pub token_ring_max_slot_version: u64,
    pub stale_tokens: u64,
    pub missing_tokens: u64,
    pub extra_tokens: u64,
    pub mismatched_tokens: u64,
    pub host_causality_edges: u64,
    pub resident_weight_bytes: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub graph_launches: u64,
    pub sync_calls: u64,
    pub kernel_launches: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaTinyDecodeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"steps\":{},\"ring_capacity\":{},\"seed_token\":{},\"vocab_size\":{},\"hidden\":{},\"last_token\":{},\"graph_replays\":{},\"graph_nodes\":{},\"observed_tokens\":{},\"observed_token_hash\":{},\"token_ring_slots_touched\":{},\"token_ring_reuses\":{},\"token_ring_max_slot_version\":{},\"stale_tokens\":{},\"missing_tokens\":{},\"extra_tokens\":{},\"mismatched_tokens\":{},\"host_causality_edges\":{},\"resident_weight_bytes\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"graph_launches\":{},\"sync_calls\":{},\"kernel_launches\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.steps,
            self.ring_capacity,
            self.seed_token,
            self.vocab_size,
            self.hidden,
            json_opt_u32(self.last_token),
            self.graph_replays,
            self.graph_nodes,
            self.observed_tokens,
            self.observed_token_hash,
            self.token_ring_slots_touched,
            self.token_ring_reuses,
            self.token_ring_max_slot_version,
            self.stale_tokens,
            self.missing_tokens,
            self.extra_tokens,
            self.mismatched_tokens,
            self.host_causality_edges,
            self.resident_weight_bytes,
            self.device_arena_bytes,
            self.pinned_host_bytes,
            self.h2d_bytes,
            self.d2h_bytes,
            self.graph_launches,
            self.sync_calls,
            self.kernel_launches,
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
    }

    fn unavailable(
        steps: u32,
        ring_capacity: u32,
        seed_token: u32,
        error: impl Into<String>,
    ) -> Self {
        Self::empty(
            SmokeStatus::Unavailable,
            steps,
            ring_capacity,
            seed_token,
            error,
        )
    }

    fn failed(steps: u32, ring_capacity: u32, seed_token: u32, error: impl Into<String>) -> Self {
        Self::empty(SmokeStatus::Failed, steps, ring_capacity, seed_token, error)
    }

    fn empty(
        status: SmokeStatus,
        steps: u32,
        ring_capacity: u32,
        seed_token: u32,
        error: impl Into<String>,
    ) -> Self {
        Self {
            status,
            steps,
            ring_capacity,
            seed_token,
            vocab_size: 4,
            hidden: 2,
            last_token: None,
            graph_replays: 0,
            graph_nodes: 0,
            observed_tokens: 0,
            observed_token_hash: 0,
            token_ring_slots_touched: 0,
            token_ring_reuses: 0,
            token_ring_max_slot_version: 0,
            stale_tokens: 0,
            missing_tokens: steps as u64,
            extra_tokens: 0,
            mismatched_tokens: 0,
            host_causality_edges: 0,
            resident_weight_bytes: 0,
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            h2d_bytes: 0,
            d2h_bytes: 0,
            graph_launches: 0,
            sync_calls: 0,
            kernel_launches: 0,
            hot_path_allocations: 0,
            error: Some(error.into()),
        }
    }
}

pub fn tiny_decode_smoke(steps: u32, ring_capacity: u32, seed_token: u32) -> CudaTinyDecodeSummary {
    let mut out = NervaCudaTinyDecodeResult::default();
    let return_code =
        unsafe { nerva_cuda_tiny_decode_smoke(steps, ring_capacity, seed_token, &mut out) };

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

fn json_opt_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_opt_str(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("\"{}\"", escape_json(value)),
        None => "null".to_string(),
    }
}
