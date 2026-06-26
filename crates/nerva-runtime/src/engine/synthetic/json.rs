use nerva_core::types::id::TokenId;

use crate::engine::synthetic::summary::{SyntheticDecodeStatus, SyntheticDecodeSummary};

impl SyntheticDecodeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            SyntheticDecodeStatus::Ok => "ok",
            SyntheticDecodeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"steps\":{},\"token_ring_capacity\":{},\"token_ring_slots_touched\":{},\"token_ring_reuses\":{},\"token_ring_max_slot_version\":{},\"seed_token\":{},\"last_token\":{},\"graph_replays\":{},\"graph_replay_events\":{},\"kernel_events\":{},\"device_events\":{},\"copy_events\":{},\"host_wait_events\":{},\"soft_visibility_syncs\":{},\"device_timeline_active_ns\":{},\"device_timeline_idle_ns\":{},\"graph_replay_latency_ns\":{},\"device_latency_ns\":{},\"copy_latency_ns\":{},\"host_wait_latency_ns\":{},\"soft_visibility_sync_latency_ns\":{},\"estimated_events\":{},\"estimated_latency_ns\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"observed_tokens\":{},\"observed_token_hash\":{},\"stale_tokens\":{},\"missing_tokens\":{},\"extra_tokens\":{},\"mismatched_tokens\":{},\"host_causality_edges\":{},\"error\":{}}}",
            status,
            self.steps,
            self.token_ring_capacity,
            self.token_ring_slots_touched,
            self.token_ring_reuses,
            self.token_ring_max_slot_version,
            self.seed_token.0,
            json_opt_token(self.last_token),
            self.graph_replays,
            self.graph_replay_events,
            self.kernel_events,
            self.device_events,
            self.copy_events,
            self.host_wait_events,
            self.soft_visibility_syncs,
            self.device_timeline_active_ns,
            self.device_timeline_idle_ns,
            self.graph_replay_latency_ns,
            self.device_latency_ns,
            self.copy_latency_ns,
            self.host_wait_latency_ns,
            self.soft_visibility_sync_latency_ns,
            self.estimated_events,
            self.estimated_latency_ns,
            self.total_latency_ns,
            self.hot_path_allocations,
            self.observed_tokens,
            self.observed_token_hash,
            self.stale_tokens,
            self.missing_tokens,
            self.extra_tokens,
            self.mismatched_tokens,
            self.host_causality_edges,
            json_opt_static_str(self.error),
        )
    }
}

fn json_opt_token(value: Option<TokenId>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.0.to_string())
}

fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{value}\""))
}
