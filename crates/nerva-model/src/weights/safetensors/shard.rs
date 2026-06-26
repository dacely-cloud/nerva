use nerva_core::types::dtype::DType;
use nerva_core::types::memory::MemoryTier;

use crate::common::json::format::{json_opt_str, json_opt_usize};
use crate::weights::layout::entry::WeightBlockRole;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SafetensorsShardHeader<'a> {
    pub file_name: &'a str,
    pub header_json: &'a str,
}

impl<'a> SafetensorsShardHeader<'a> {
    pub const fn new(file_name: &'a str, header_json: &'a str) -> Self {
        Self {
            file_name,
            header_json,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SafetensorsShardPlanEntry {
    pub tensor_name: String,
    pub shard_file: String,
    pub role: WeightBlockRole,
    pub layer: Option<u32>,
    pub dtype: DType,
    pub tier: MemoryTier,
    pub bytes: usize,
    pub data_offset_begin: usize,
    pub data_offset_end: usize,
    pub file_offset_begin: usize,
    pub file_offset_end: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SafetensorsShardPlanShard {
    pub file_name: String,
    pub tensor_count: usize,
    pub payload_bytes: usize,
    pub header_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SafetensorsShardPlan {
    pub entries: Vec<SafetensorsShardPlanEntry>,
    pub shards: Vec<SafetensorsShardPlanShard>,
    pub total_weight_bytes: usize,
    pub index_total_size: Option<usize>,
    pub manifest_hash: u64,
    pub index_hash: u64,
    pub plan_hash: u64,
}

impl SafetensorsShardPlan {
    pub fn to_json(&self) -> String {
        let first = self.entries.first().map(|entry| entry.tensor_name.as_str());
        let last = self.entries.last().map(|entry| entry.tensor_name.as_str());
        format!(
            "{{\"entries\":{},\"shards\":{},\"total_weight_bytes\":{},\"index_total_size\":{},\"first_tensor\":{},\"last_tensor\":{},\"manifest_hash\":{},\"index_hash\":{},\"plan_hash\":{}}}",
            self.entries.len(),
            self.shards.len(),
            self.total_weight_bytes,
            json_opt_usize(self.index_total_size),
            json_opt_str(first),
            json_opt_str(last),
            self.manifest_hash,
            self.index_hash,
            self.plan_hash,
        )
    }
}
