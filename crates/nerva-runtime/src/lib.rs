#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use nerva_core::{DeviceOrdinal, Result, ensure_supported_linux_host};
use nerva_ledger::TokenLedger;

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
}
