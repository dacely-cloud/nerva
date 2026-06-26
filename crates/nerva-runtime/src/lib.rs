#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use nerva_core::{DeviceOrdinal, MemoryTier, Result, ensure_supported_linux_host};
use nerva_ledger::TokenLedger;
use nerva_memory::BlockRegistry;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    pub device: DeviceOrdinal,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            device: DeviceOrdinal(0),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ResidencyBudget {
    pub vram_bytes: usize,
    pub pinned_dram_bytes: usize,
    pub dram_bytes: usize,
}

impl ResidencyBudget {
    pub const fn new(vram_bytes: usize, pinned_dram_bytes: usize, dram_bytes: usize) -> Self {
        Self {
            vram_bytes,
            pinned_dram_bytes,
            dram_bytes,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Runtime {
    config: RuntimeConfig,
}

impl Runtime {
    pub fn new(config: RuntimeConfig) -> Result<Self> {
        ensure_supported_linux_host()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> RuntimeConfig {
        self.config
    }

    pub fn empty_token_ledger(&self, token_index: u64) -> TokenLedger {
        let _ = self.config;
        TokenLedger::new(token_index)
    }

    pub fn block_registry(&self, budget: ResidencyBudget) -> BlockRegistry {
        let _ = self.config;
        BlockRegistry::new([
            (MemoryTier::Vram, budget.vram_bytes),
            (MemoryTier::PinnedDram, budget.pinned_dram_bytes),
            (MemoryTier::Dram, budget.dram_bytes),
        ])
    }
}

pub fn cuda_smoke() -> nerva_cuda::CudaSmokeSummary {
    nerva_cuda::smoke()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_uses_device_zero_by_default() {
        let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
        assert_eq!(runtime.config().device, DeviceOrdinal(0));
        assert_eq!(runtime.empty_token_ledger(9).token_index, 9);
    }

    #[test]
    fn runtime_creates_residency_registry_from_budget() {
        let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
        let registry = runtime.block_registry(ResidencyBudget::new(1024, 2048, 4096));
        assert_eq!(registry.remaining_bytes(MemoryTier::Vram), Some(1024));
        assert_eq!(registry.remaining_bytes(MemoryTier::PinnedDram), Some(2048));
        assert_eq!(registry.remaining_bytes(MemoryTier::Dram), Some(4096));
    }
}
