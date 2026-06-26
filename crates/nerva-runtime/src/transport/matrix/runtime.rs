use nerva_core::types::error::Result;

use crate::engine::runtime::Runtime;
use crate::transport::matrix::run as matrix_run;
use crate::transport::matrix::types::TransportCapabilityMatrixSummary;

impl Runtime {
    pub fn run_transport_capability_matrix_probe(
        &self,
    ) -> Result<TransportCapabilityMatrixSummary> {
        let capabilities = self.discover_capabilities();
        matrix_run::run_transport_capability_matrix_probe(self.config.device, &capabilities)
    }
}
