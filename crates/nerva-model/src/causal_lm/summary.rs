use nerva_core::types::dtype::DType;
use nerva_core::types::id::token::TokenId;

use crate::common::token::token_ids_to_json;
use crate::precision::bits::dtype_label;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfCausalLmSmokeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfCausalLmSmokeSummary {
    pub status: HfCausalLmSmokeStatus,
    pub dtype: DType,
    pub layers: usize,
    pub hidden: usize,
    pub vocab_size: usize,
    pub manifest_entries: usize,
    pub shard_plan_entries: usize,
    pub tensors_loaded: usize,
    pub bytes_loaded: usize,
    pub final_norm_loaded: bool,
    pub tied_lm_head: bool,
    pub steps: usize,
    pub tokens: Vec<TokenId>,
    pub expected_tokens: Vec<TokenId>,
    pub parity: bool,
    pub ledger_count: u64,
    pub cpu_events: u64,
    pub execution_decisions: u64,
    pub output_hash: u64,
    pub data_hash: u64,
    pub hot_path_allocations: u64,
}

impl HfCausalLmSmokeSummary {
    pub fn passed(&self) -> bool {
        self.parity
            && self.final_norm_loaded
            && self.manifest_entries == self.shard_plan_entries
            && self.tensors_loaded == self.manifest_entries
            && self.ledger_count == self.steps as u64
            && self.cpu_events == self.steps as u64
            && self.execution_decisions == self.steps as u64
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            HfCausalLmSmokeStatus::Ok => "ok",
        };
        let dtype = dtype_label(self.dtype).unwrap_or("unsupported");
        format!(
            "{{\"status\":\"{}\",\"dtype\":\"{}\",\"layers\":{},\"hidden\":{},\"vocab_size\":{},\"manifest_entries\":{},\"shard_plan_entries\":{},\"tensors_loaded\":{},\"bytes_loaded\":{},\"final_norm_loaded\":{},\"tied_lm_head\":{},\"steps\":{},\"tokens\":{},\"expected_tokens\":{},\"parity\":{},\"ledger_count\":{},\"cpu_events\":{},\"execution_decisions\":{},\"output_hash\":{},\"data_hash\":{},\"hot_path_allocations\":{}}}",
            status,
            dtype,
            self.layers,
            self.hidden,
            self.vocab_size,
            self.manifest_entries,
            self.shard_plan_entries,
            self.tensors_loaded,
            self.bytes_loaded,
            self.final_norm_loaded,
            self.tied_lm_head,
            self.steps,
            token_ids_to_json(&self.tokens),
            token_ids_to_json(&self.expected_tokens),
            self.parity,
            self.ledger_count,
            self.cpu_events,
            self.execution_decisions,
            self.output_hash,
            self.data_hash,
            self.hot_path_allocations,
        )
    }
}
