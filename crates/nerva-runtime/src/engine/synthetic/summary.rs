use nerva_core::types::id::token::TokenId;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SyntheticDecodeStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SyntheticDecodeSummary {
    pub status: SyntheticDecodeStatus,
    pub steps: u64,
    pub token_ring_capacity: usize,
    pub token_ring_slots_touched: u64,
    pub token_ring_reuses: u64,
    pub token_ring_max_slot_version: u64,
    pub seed_token: TokenId,
    pub last_token: Option<TokenId>,
    pub graph_replays: u64,
    pub graph_replay_events: u64,
    pub kernel_events: u64,
    pub device_events: u64,
    pub copy_events: u64,
    pub host_wait_events: u64,
    pub soft_visibility_syncs: u64,
    pub device_timeline_active_ns: u64,
    pub device_timeline_idle_ns: u64,
    pub graph_replay_latency_ns: u64,
    pub device_latency_ns: u64,
    pub copy_latency_ns: u64,
    pub host_wait_latency_ns: u64,
    pub soft_visibility_sync_latency_ns: u64,
    pub estimated_events: u64,
    pub estimated_latency_ns: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub observed_tokens: u64,
    pub observed_token_hash: u64,
    pub stale_tokens: u64,
    pub missing_tokens: u64,
    pub extra_tokens: u64,
    pub mismatched_tokens: u64,
    pub host_causality_edges: u64,
    pub error: Option<&'static str>,
}
