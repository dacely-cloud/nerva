use nerva_core::types::error::Result;
use nerva_memory::security::probe::run_security_isolation_probe;
use nerva_memory::security::summary::SecurityIsolationSummary;

use crate::engine::runtime::Runtime;

impl Runtime {
    pub fn run_security_isolation_probe(&self) -> Result<SecurityIsolationSummary> {
        let _ = self.config;
        run_security_isolation_probe()
    }
}
