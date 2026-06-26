use nerva_core::types::arch::ensure_supported_linux_host;
use nerva_core::types::error::Result;
use nerva_core::types::id::DeviceOrdinal;
use nerva_ledger::types::token::TokenLedger;

use crate::capabilities::discovery::discover_capabilities;
use crate::capabilities::snapshot::{CapabilitySnapshot, TopologySnapshot};
use crate::capabilities::topology::discover_topology_snapshot;
use crate::token::SyntheticEngine;

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

#[derive(Clone, Debug)]
pub struct Runtime {
    pub(crate) config: RuntimeConfig,
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

    pub fn synthetic_engine(&self, token_ring_capacity: usize) -> Result<SyntheticEngine> {
        SyntheticEngine::new(token_ring_capacity, self.config.device)
    }

    pub fn discover_topology(&self) -> TopologySnapshot {
        let _ = self.config;
        discover_topology_snapshot()
    }

    pub fn discover_capabilities(&self) -> CapabilitySnapshot {
        let _ = self.config;
        discover_capabilities()
    }
}
