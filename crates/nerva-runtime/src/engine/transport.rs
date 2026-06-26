use nerva_core::types::error::Result;

use crate::engine::runtime::Runtime;
use crate::transport::matrix::{self, TransportCapabilityMatrixSummary};
use crate::transport::path::{self, TransportPathDecision, TransportPathRequest};
use crate::transport::probe::{self, TransportPathProbeSummary};

impl Runtime {
    pub fn plan_transport_path(
        &self,
        request: TransportPathRequest,
    ) -> Result<TransportPathDecision> {
        let _ = self.config;
        path::plan_transport_path(request)
    }

    pub fn run_transport_path_probe(&self) -> Result<TransportPathProbeSummary> {
        let capabilities = self.discover_capabilities();
        probe::run_transport_path_probe(self.config.device, &capabilities)
    }

    pub fn run_transport_capability_matrix_probe(
        &self,
    ) -> Result<TransportCapabilityMatrixSummary> {
        let capabilities = self.discover_capabilities();
        matrix::run_transport_capability_matrix_probe(self.config.device, &capabilities)
    }
}
