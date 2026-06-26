use crate::warm_compute::strategy::WarmComputeStrategy;

#[derive(Clone, Debug, PartialEq)]
pub struct WarmComputeCandidate {
    pub strategy: WarmComputeStrategy,
    pub visible_ns: u64,
    pub output_hash: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WarmComputeProbeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WarmComputeProbeSummary {
    pub status: WarmComputeProbeStatus,
    pub rows: usize,
    pub cols: usize,
    pub candidates: Vec<WarmComputeCandidate>,
    pub selected_strategy: WarmComputeStrategy,
    pub parity: bool,
    pub cpu_beats_staged: bool,
    pub execution_decisions: u64,
    pub cpu_events: u64,
    pub device_events: u64,
    pub copy_events: u64,
    pub copy_bytes: usize,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub output_hash: u64,
}

impl WarmComputeProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            WarmComputeProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"rows\":{},\"cols\":{},\"selected_strategy\":\"{}\",\"parity\":{},\"cpu_beats_staged\":{},\"candidate_count\":{},\"execution_decisions\":{},\"cpu_events\":{},\"device_events\":{},\"copy_events\":{},\"copy_bytes\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"output_hash\":{}}}",
            status,
            self.rows,
            self.cols,
            self.selected_strategy.label(),
            self.parity,
            self.cpu_beats_staged,
            self.candidates.len(),
            self.execution_decisions,
            self.cpu_events,
            self.device_events,
            self.copy_events,
            self.copy_bytes,
            self.total_latency_ns,
            self.hot_path_allocations,
            self.output_hash,
        )
    }
}
