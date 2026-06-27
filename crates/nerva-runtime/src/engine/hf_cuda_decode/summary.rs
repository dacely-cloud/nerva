use nerva_core::types::id::token::TokenId;
use nerva_cuda::smoke::status::SmokeStatus;

use crate::engine::hf_cuda_decode::hash::tokens_json;

#[derive(Clone, Debug)]
pub struct HfCudaSeedDecodeSummary {
    pub status: SmokeStatus,
    pub steps_requested: usize,
    pub tokens: Vec<TokenId>,
    pub expected_tokens: Vec<TokenId>,
    pub parity: bool,
    pub ledger_count: u64,
    pub device_events: u64,
    pub copy_events: u64,
    pub hard_syncs: u64,
    pub execution_decisions: u64,
    pub resident_weight_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub graph_replays: u64,
    pub graph_nodes: u64,
    pub graph_launches: u64,
    pub graph_replay_events: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub host_causality_edges: u64,
    pub hot_path_allocations: u64,
    pub output_hash: u64,
    pub expected_hash: u64,
    pub error: Option<String>,
}

impl HfCudaSeedDecodeSummary {
    pub fn passed(&self) -> bool {
        self.status == SmokeStatus::Ok && self.parity && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"status\":\"{}\",\"steps_requested\":{},\"tokens\":{},\"expected_tokens\":{},\"parity\":{},\"ledger_count\":{},\"device_events\":{},\"copy_events\":{},\"hard_syncs\":{},\"execution_decisions\":{},\"resident_weight_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"graph_replays\":{},\"graph_nodes\":{},\"graph_launches\":{},\"graph_replay_events\":{},\"kernel_launches\":{},\"sync_calls\":{},\"host_causality_edges\":{},\"hot_path_allocations\":{},\"output_hash\":{},\"expected_hash\":{},\"error\":{}}}",
            status_json(&self.status),
            self.steps_requested,
            tokens_json(&self.tokens),
            tokens_json(&self.expected_tokens),
            self.parity,
            self.ledger_count,
            self.device_events,
            self.copy_events,
            self.hard_syncs,
            self.execution_decisions,
            self.resident_weight_bytes,
            self.h2d_bytes,
            self.d2h_bytes,
            self.graph_replays,
            self.graph_nodes,
            self.graph_launches,
            self.graph_replay_events,
            self.kernel_launches,
            self.sync_calls,
            self.host_causality_edges,
            self.hot_path_allocations,
            self.output_hash,
            self.expected_hash,
            json_string(self.error.as_deref()),
        )
    }
}

fn status_json(status: &SmokeStatus) -> &'static str {
    match status {
        SmokeStatus::Ok => "ok",
        SmokeStatus::Unavailable => "unavailable",
        SmokeStatus::Failed => "failed",
    }
}

fn json_string(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\"")),
        None => "null".to_string(),
    }
}
