use nerva_core::types::{BlockKind, DType, MemoryTier, NervaError, Result};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelContractKind {
    DecodeGraph,
    DenseMatvec,
    BlockwiseAttention,
    Sampler,
    ResidencyTransfer,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelBufferRole {
    Input,
    Output,
    InOut,
    Scratch,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct LaunchBounds {
    pub max_grid_blocks: u32,
    pub max_threads_per_block: u32,
}

impl LaunchBounds {
    pub fn new(max_grid_blocks: u32, max_threads_per_block: u32) -> Result<Self> {
        if max_grid_blocks == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "kernel launch must allow at least one grid block".to_string(),
            });
        }
        if max_threads_per_block == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "kernel launch must allow at least one thread per block".to_string(),
            });
        }
        Ok(Self {
            max_grid_blocks,
            max_threads_per_block,
        })
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KernelBufferContract {
    pub name: &'static str,
    pub role: KernelBufferRole,
    pub block_kind: BlockKind,
    pub dtype: DType,
    pub expected_tier: MemoryTier,
    pub min_bytes: usize,
}

impl KernelBufferContract {
    pub fn new(
        name: &'static str,
        role: KernelBufferRole,
        block_kind: BlockKind,
        dtype: DType,
        expected_tier: MemoryTier,
        min_bytes: usize,
    ) -> Result<Self> {
        if name.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "kernel buffer contract name must be non-empty".to_string(),
            });
        }
        if min_bytes == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "kernel buffer contract must require non-zero bytes".to_string(),
            });
        }
        Ok(Self {
            name,
            role,
            block_kind,
            dtype,
            expected_tier,
            min_bytes,
        })
    }

    pub const fn requires_device_residency(self) -> bool {
        matches!(
            self.expected_tier,
            MemoryTier::Vram | MemoryTier::SharedHbmOrLpddr
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KernelContract {
    pub name: &'static str,
    pub kind: KernelContractKind,
    pub launch_bounds: LaunchBounds,
    pub buffers: Vec<KernelBufferContract>,
    pub hot_path_allocation_allowed: bool,
}

impl KernelContract {
    pub fn new(
        name: &'static str,
        kind: KernelContractKind,
        launch_bounds: LaunchBounds,
        buffers: Vec<KernelBufferContract>,
    ) -> Result<Self> {
        if name.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "kernel contract name must be non-empty".to_string(),
            });
        }
        if buffers.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "kernel contract must describe at least one buffer".to_string(),
            });
        }
        Ok(Self {
            name,
            kind,
            launch_bounds,
            buffers,
            hot_path_allocation_allowed: false,
        })
    }

    pub fn with_hot_path_allocation_allowed(mut self, allowed: bool) -> Self {
        self.hot_path_allocation_allowed = allowed;
        self
    }

    pub fn require_decode_ready(&self) -> Result<()> {
        if self.hot_path_allocation_allowed {
            return Err(NervaError::InvalidArgument {
                reason: format!("kernel contract {} permits hot-path allocation", self.name),
            });
        }
        if !self
            .buffers
            .iter()
            .any(|buffer| buffer.requires_device_residency())
        {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "kernel contract {} has no device-resident buffers",
                    self.name
                ),
            });
        }
        Ok(())
    }
}

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
