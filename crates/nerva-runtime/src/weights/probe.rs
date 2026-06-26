use nerva_core::types::id::ResidentBlockId;

use crate::weights::json::{json_opt_block_id, json_opt_string};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResidentWeightProbeStatus {
    Ok,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightProbeSummary {
    pub status: ResidentWeightProbeStatus,
    pub blocks: usize,
    pub total_weight_bytes: usize,
    pub dram_used_bytes: usize,
    pub vram_used_bytes: usize,
    pub residency_decisions: u64,
    pub first_block_id: Option<ResidentBlockId>,
    pub last_block_id: Option<ResidentBlockId>,
    pub first_tensor: Option<String>,
    pub last_tensor: Option<String>,
    pub manifest_hash: u64,
    pub hot_path_allocations: u64,
}

impl ResidentWeightProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            ResidentWeightProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"blocks\":{},\"total_weight_bytes\":{},\"dram_used_bytes\":{},\"vram_used_bytes\":{},\"residency_decisions\":{},\"first_block_id\":{},\"last_block_id\":{},\"first_tensor\":{},\"last_tensor\":{},\"manifest_hash\":{},\"hot_path_allocations\":{}}}",
            status,
            self.blocks,
            self.total_weight_bytes,
            self.dram_used_bytes,
            self.vram_used_bytes,
            self.residency_decisions,
            json_opt_block_id(self.first_block_id),
            json_opt_block_id(self.last_block_id),
            json_opt_string(self.first_tensor.as_deref()),
            json_opt_string(self.last_tensor.as_deref()),
            self.manifest_hash,
            self.hot_path_allocations,
        )
    }
}
