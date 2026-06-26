#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ComputeNearDataProbeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComputeNearDataProbeSummary {
    pub status: ComputeNearDataProbeStatus,
    pub rows: usize,
    pub cols: usize,
    pub split_row: usize,
    pub blocks: usize,
    pub dram_blocks: u64,
    pub vram_blocks: u64,
    pub output: Vec<f32>,
    pub reference: Vec<f32>,
    pub output_hash: u64,
    pub reference_hash: u64,
    pub max_abs_error: f32,
    pub parity: bool,
    pub execution_decisions: u64,
    pub runtime_timestamp_decisions: u64,
    pub measured_candidate_costs: u64,
    pub estimated_candidate_costs: u64,
    pub block_version_dependencies: u64,
    pub cpu_events: u64,
    pub device_events: u64,
    pub copy_events: u64,
    pub merge_bytes: usize,
    pub hot_path_allocations: u64,
}

impl ComputeNearDataProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            ComputeNearDataProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"rows\":{},\"cols\":{},\"split_row\":{},\"blocks\":{},\"dram_blocks\":{},\"vram_blocks\":{},\"output\":{},\"reference\":{},\"output_hash\":{},\"reference_hash\":{},\"max_abs_error\":{},\"parity\":{},\"execution_decisions\":{},\"runtime_timestamp_decisions\":{},\"measured_candidate_costs\":{},\"estimated_candidate_costs\":{},\"block_version_dependencies\":{},\"cpu_events\":{},\"device_events\":{},\"copy_events\":{},\"merge_bytes\":{},\"hot_path_allocations\":{}}}",
            status,
            self.rows,
            self.cols,
            self.split_row,
            self.blocks,
            self.dram_blocks,
            self.vram_blocks,
            json_f32_array(&self.output),
            json_f32_array(&self.reference),
            self.output_hash,
            self.reference_hash,
            self.max_abs_error,
            self.parity,
            self.execution_decisions,
            self.runtime_timestamp_decisions,
            self.measured_candidate_costs,
            self.estimated_candidate_costs,
            self.block_version_dependencies,
            self.cpu_events,
            self.device_events,
            self.copy_events,
            self.merge_bytes,
            self.hot_path_allocations,
        )
    }
}

fn json_f32_array(values: &[f32]) -> String {
    let items = values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}
