use nerva_core::types::block::BlockKind;
use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;
use nerva_core::types::memory::MemoryTier;

use crate::contract::types::{
    KernelBufferContract, KernelBufferRole, KernelContract, KernelContractKind, LaunchBounds,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelContractProbeStatus {
    Ok,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelContractProbeSummary {
    pub status: KernelContractProbeStatus,
    pub contract_count: usize,
    pub buffer_count: usize,
    pub device_resident_buffers: usize,
    pub hot_path_allocation_allowed: bool,
    pub max_grid_blocks: u32,
    pub max_threads_per_block: u32,
}

impl KernelContractProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            KernelContractProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"contract_count\":{},\"buffer_count\":{},\"device_resident_buffers\":{},\"hot_path_allocation_allowed\":{},\"max_grid_blocks\":{},\"max_threads_per_block\":{}}}",
            status,
            self.contract_count,
            self.buffer_count,
            self.device_resident_buffers,
            self.hot_path_allocation_allowed,
            self.max_grid_blocks,
            self.max_threads_per_block,
        )
    }
}

pub fn kernel_contract_probe() -> Result<KernelContractProbeSummary> {
    let bounds = LaunchBounds::new(64, 256)?;
    let token_ring = KernelBufferContract::new(
        "device_token_ring",
        KernelBufferRole::InOut,
        BlockKind::TokenState,
        DType::U32,
        MemoryTier::Vram,
        4096,
    )?;
    let logits = KernelBufferContract::new(
        "device_logits",
        KernelBufferRole::Output,
        BlockKind::Logits,
        DType::F32,
        MemoryTier::Vram,
        4096,
    )?;
    let contract = KernelContract::new(
        "synthetic_decode",
        KernelContractKind::DecodeGraph,
        bounds,
        vec![token_ring, logits],
    )?;
    contract.require_decode_ready()?;

    Ok(KernelContractProbeSummary {
        status: KernelContractProbeStatus::Ok,
        contract_count: 1,
        buffer_count: contract.buffers.len(),
        device_resident_buffers: contract
            .buffers
            .iter()
            .filter(|buffer| buffer.requires_device_residency())
            .count(),
        hot_path_allocation_allowed: contract.hot_path_allocation_allowed,
        max_grid_blocks: contract.launch_bounds.max_grid_blocks,
        max_threads_per_block: contract.launch_bounds.max_threads_per_block,
    })
}
