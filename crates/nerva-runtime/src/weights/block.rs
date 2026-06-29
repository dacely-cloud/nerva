use nerva_core::types::dtype::DType;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::registry::table::registry::BlockRegistry;
use nerva_model::weights::layout::entry::WeightBlockRole;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightBlockRef {
    pub name: String,
    pub block_id: ResidentBlockId,
    pub role: WeightBlockRole,
    pub bytes: usize,
    pub dtype: DType,
    pub tier: MemoryTier,
    pub source_shard: Option<String>,
    pub file_offset_begin: Option<usize>,
    pub file_offset_end: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentWeightTable {
    pub registry: BlockRegistry,
    pub entries: Vec<ResidentWeightBlockRef>,
    pub total_weight_bytes: usize,
    pub manifest_hash: u64,
    pub ledger: TokenLedger,
}
