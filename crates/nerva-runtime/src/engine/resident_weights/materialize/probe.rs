use nerva_core::types::error::Result;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_model::weights::manifest::hf_tensor_manifest_probe;

use crate::engine::runtime::Runtime;
use crate::weights::probe::{ResidentWeightProbeStatus, ResidentWeightProbeSummary};

impl Runtime {
    pub fn run_resident_weight_probe(&self) -> Result<ResidentWeightProbeSummary> {
        let manifest = hf_tensor_manifest_probe()?.manifest;
        let table = self.materialize_hf_weight_manifest(&manifest)?;
        let first = table.entries.first();
        let last = table.entries.last();

        Ok(ResidentWeightProbeSummary {
            status: ResidentWeightProbeStatus::Ok,
            blocks: table.entries.len(),
            total_weight_bytes: table.total_weight_bytes,
            dram_used_bytes: table.registry.used_bytes(MemoryTier::Dram),
            vram_used_bytes: table.registry.used_bytes(MemoryTier::Vram),
            residency_decisions: table.ledger.residency_decisions.len() as u64,
            first_block_id: first.map(|entry| entry.block_id),
            last_block_id: last.map(|entry| entry.block_id),
            first_tensor: first.map(|entry| entry.name.clone()),
            last_tensor: last.map(|entry| entry.name.clone()),
            manifest_hash: table.manifest_hash,
            hot_path_allocations: table.ledger.hot_path_allocations,
        })
    }
}
