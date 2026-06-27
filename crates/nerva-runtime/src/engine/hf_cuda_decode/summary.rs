use nerva_core::types::id::token::TokenId;
use nerva_cuda::smoke::status::SmokeStatus;

use crate::engine::hf_cuda_decode::hash::tokens_json;

#[derive(Clone, Debug, Default)]
pub struct HfCudaResidentWeightSummary {
    pub plan_steps: u64,
    pub plan_weight_bytes: u64,
    pub plan_gpu_resident_steps: u64,
    pub plan_gpu_staged_steps: u64,
    pub plan_fallback_steps: u64,
    pub plan_block_version_dependencies: u64,
    pub run_steps: u64,
    pub run_gpu_resident_steps: u64,
    pub run_gpu_staged_steps: u64,
    pub run_fallback_steps: u64,
    pub run_block_version_dependencies: u64,
    pub hot_path_allocations: u64,
}

impl HfCudaResidentWeightSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"plan_steps\":{},\"plan_weight_bytes\":{},\"plan_gpu_resident_steps\":{},\"plan_gpu_staged_steps\":{},\"plan_fallback_steps\":{},\"plan_block_version_dependencies\":{},\"run_steps\":{},\"run_gpu_resident_steps\":{},\"run_gpu_staged_steps\":{},\"run_fallback_steps\":{},\"run_block_version_dependencies\":{},\"hot_path_allocations\":{}}}",
            self.plan_steps,
            self.plan_weight_bytes,
            self.plan_gpu_resident_steps,
            self.plan_gpu_staged_steps,
            self.plan_fallback_steps,
            self.plan_block_version_dependencies,
            self.run_steps,
            self.run_gpu_resident_steps,
            self.run_gpu_staged_steps,
            self.run_fallback_steps,
            self.run_block_version_dependencies,
            self.hot_path_allocations,
        )
    }
}

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
    pub resident_kv_bytes: u64,
    pub kv_tokens: u64,
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
    pub resident_weights: HfCudaResidentWeightSummary,
    pub error: Option<String>,
}

impl HfCudaSeedDecodeSummary {
    pub fn passed(&self) -> bool {
        self.status == SmokeStatus::Ok && self.parity && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"status\":\"{}\",\"steps_requested\":{},\"tokens\":{},\"expected_tokens\":{},\"parity\":{},\"ledger_count\":{},\"device_events\":{},\"copy_events\":{},\"hard_syncs\":{},\"execution_decisions\":{},\"resident_weight_bytes\":{},\"resident_kv_bytes\":{},\"kv_tokens\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"graph_replays\":{},\"graph_nodes\":{},\"graph_launches\":{},\"graph_replay_events\":{},\"kernel_launches\":{},\"sync_calls\":{},\"host_causality_edges\":{},\"hot_path_allocations\":{},\"output_hash\":{},\"expected_hash\":{},\"resident_weight_plan\":{},\"error\":{}}}",
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
            self.resident_kv_bytes,
            self.kv_tokens,
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
            self.resident_weights.to_json(),
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
