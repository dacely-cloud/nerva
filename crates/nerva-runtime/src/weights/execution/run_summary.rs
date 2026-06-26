use nerva_ledger::types::token::ledger::TokenLedger;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightExecutionRunSummary {
    pub steps: usize,
    pub total_weight_bytes: usize,
    pub total_latency_ns: u64,
    pub cpu_events: u64,
    pub device_events: u64,
    pub copy_events: u64,
    pub gpu_resident_steps: u64,
    pub gpu_staged_steps: u64,
    pub fallback_steps: u64,
    pub fallback_decisions: u64,
    pub block_version_dependencies: u64,
    pub hot_path_allocations: u64,
    pub ledger: TokenLedger,
}

impl ResidentWeightExecutionRunSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"steps\":{},\"total_weight_bytes\":{},\"total_latency_ns\":{},\"cpu_events\":{},\"device_events\":{},\"copy_events\":{},\"gpu_resident_steps\":{},\"gpu_staged_steps\":{},\"fallback_steps\":{},\"fallback_decisions\":{},\"block_version_dependencies\":{},\"hot_path_allocations\":{}}}",
            self.steps,
            self.total_weight_bytes,
            self.total_latency_ns,
            self.cpu_events,
            self.device_events,
            self.copy_events,
            self.gpu_resident_steps,
            self.gpu_staged_steps,
            self.fallback_steps,
            self.fallback_decisions,
            self.block_version_dependencies,
            self.hot_path_allocations,
        )
    }
}
