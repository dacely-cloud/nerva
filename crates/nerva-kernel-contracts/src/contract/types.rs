use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;

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
