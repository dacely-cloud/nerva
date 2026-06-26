use crate::weights::json::json_opt_string;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightHotsetSummary {
    pub promoted_blocks: usize,
    pub promoted_bytes: usize,
    pub dram_used_bytes: usize,
    pub vram_used_bytes: usize,
    pub residency_decisions: u64,
    pub first_promoted_tensor: Option<String>,
    pub last_promoted_tensor: Option<String>,
    pub hot_path_allocations: u64,
}

impl ResidentWeightHotsetSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"promoted_blocks\":{},\"promoted_bytes\":{},\"dram_used_bytes\":{},\"vram_used_bytes\":{},\"residency_decisions\":{},\"first_promoted_tensor\":{},\"last_promoted_tensor\":{},\"hot_path_allocations\":{}}}",
            self.promoted_blocks,
            self.promoted_bytes,
            self.dram_used_bytes,
            self.vram_used_bytes,
            self.residency_decisions,
            json_opt_string(self.first_promoted_tensor.as_deref()),
            json_opt_string(self.last_promoted_tensor.as_deref()),
            self.hot_path_allocations,
        )
    }
}
