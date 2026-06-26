#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TieredKvAttentionProbeStatus {
    Ok,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TieredKvAttentionProbeSummary {
    pub status: TieredKvAttentionProbeStatus,
    pub pages: usize,
    pub tokens: usize,
    pub dram_pages: u64,
    pub vram_pages: u64,
    pub output: [f32; 2],
    pub reference: [f32; 2],
    pub max_abs_error: f32,
    pub parity: bool,
    pub output_hash: u64,
    pub reference_hash: u64,
    pub execution_decisions: u64,
    pub runtime_timestamp_decisions: u64,
    pub measured_candidate_costs: u64,
    pub estimated_candidate_costs: u64,
    pub block_version_dependencies: u64,
    pub cpu_block_events: u64,
    pub device_block_events: u64,
    pub hot_path_allocations: u64,
}

impl TieredKvAttentionProbeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            TieredKvAttentionProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"pages\":{},\"tokens\":{},\"dram_pages\":{},\"vram_pages\":{},\"output\":[{},{}],\"reference\":[{},{}],\"max_abs_error\":{},\"parity\":{},\"output_hash\":{},\"reference_hash\":{},\"execution_decisions\":{},\"runtime_timestamp_decisions\":{},\"measured_candidate_costs\":{},\"estimated_candidate_costs\":{},\"block_version_dependencies\":{},\"cpu_block_events\":{},\"device_block_events\":{},\"hot_path_allocations\":{}}}",
            status,
            self.pages,
            self.tokens,
            self.dram_pages,
            self.vram_pages,
            self.output[0],
            self.output[1],
            self.reference[0],
            self.reference[1],
            self.max_abs_error,
            self.parity,
            self.output_hash,
            self.reference_hash,
            self.execution_decisions,
            self.runtime_timestamp_decisions,
            self.measured_candidate_costs,
            self.estimated_candidate_costs,
            self.block_version_dependencies,
            self.cpu_block_events,
            self.device_block_events,
            self.hot_path_allocations,
        )
    }
}
