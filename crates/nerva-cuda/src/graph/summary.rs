use crate::json::{json_opt_str, json_opt_u32};
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaSyntheticGraphSummary {
    pub status: SmokeStatus,
    pub steps: u32,
    pub ring_capacity: u32,
    pub seed_token: u32,
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
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub graph_launches: u64,
    pub sync_calls: u64,
    pub d2h_bytes: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaSyntheticGraphSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"steps\":{},\"ring_capacity\":{},\"seed_token\":{},\"last_token\":{},\"graph_replays\":{},\"graph_nodes\":{},\"observed_tokens\":{},\"observed_token_hash\":{},\"token_ring_slots_touched\":{},\"token_ring_reuses\":{},\"token_ring_max_slot_version\":{},\"stale_tokens\":{},\"missing_tokens\":{},\"extra_tokens\":{},\"mismatched_tokens\":{},\"host_causality_edges\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"graph_launches\":{},\"sync_calls\":{},\"D2H_bytes\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.steps,
            self.ring_capacity,
            self.seed_token,
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
            self.device_arena_bytes,
            self.pinned_host_bytes,
            self.graph_launches,
            self.sync_calls,
            self.d2h_bytes,
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
    }

    pub(crate) fn unavailable(
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

    pub(crate) fn failed(
        steps: u32,
        ring_capacity: u32,
        seed_token: u32,
        error: impl Into<String>,
    ) -> Self {
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
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            graph_launches: 0,
            sync_calls: 0,
            d2h_bytes: 0,
            hot_path_allocations: 0,
            error: Some(error.into()),
        }
    }
}
