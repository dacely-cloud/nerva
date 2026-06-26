use crate::weights::json::json_opt_string;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightHotsetSummary {
    pub considered_blocks: usize,
    pub promoted_blocks: usize,
    pub promoted_bytes: usize,
    pub kept_dram_blocks: usize,
    pub budget_limited_blocks: usize,
    pub capacity_limited_blocks: usize,
    pub already_hot_blocks: usize,
    pub dram_used_bytes: usize,
    pub vram_used_bytes: usize,
    pub residency_decisions: u64,
    pub first_promoted_tensor: Option<String>,
    pub last_promoted_tensor: Option<String>,
    pub last_keep_reason: Option<&'static str>,
    pub hot_path_allocations: u64,
}

impl ResidentWeightHotsetSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"considered_blocks\":{},\"promoted_blocks\":{},\"promoted_bytes\":{},\"kept_dram_blocks\":{},\"budget_limited_blocks\":{},\"capacity_limited_blocks\":{},\"already_hot_blocks\":{},\"dram_used_bytes\":{},\"vram_used_bytes\":{},\"residency_decisions\":{},\"first_promoted_tensor\":{},\"last_promoted_tensor\":{},\"last_keep_reason\":{},\"hot_path_allocations\":{}}}",
            self.considered_blocks,
            self.promoted_blocks,
            self.promoted_bytes,
            self.kept_dram_blocks,
            self.budget_limited_blocks,
            self.capacity_limited_blocks,
            self.already_hot_blocks,
            self.dram_used_bytes,
            self.vram_used_bytes,
            self.residency_decisions,
            json_opt_string(self.first_promoted_tensor.as_deref()),
            json_opt_string(self.last_promoted_tensor.as_deref()),
            json_opt_string(self.last_keep_reason),
            self.hot_path_allocations,
        )
    }
}
