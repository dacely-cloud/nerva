use nerva_core::types::error::Result;

use crate::engine::runtime::Runtime;
use crate::transport::fabric::backend::{FabricBackendSummary, run_fabric_backend_probe};
use crate::transport::fabric::probe::run_fabric_topology_probe;
use crate::transport::fabric::summary::FabricTopologySummary;
use crate::transport::matrix::run as matrix_run;
use crate::transport::matrix::types::TransportCapabilityMatrixSummary;
use crate::transport::path::{self, TransportPathDecision, TransportPathRequest};
use crate::transport::probe::{self, TransportPathProbeSummary};
use crate::transport::registration::probe::run_transport_registration_probe;
use crate::transport::registration::summary::TransportRegistrationSummary;
use crate::transport::stage::config::StagePipelineConfig;
use crate::transport::stage::run;
use crate::transport::stage::summary::StagePipelineSummary;

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

    pub fn run_fabric_topology_probe(&self) -> FabricTopologySummary {
        let capabilities = self.discover_capabilities();
        run_fabric_topology_probe(&capabilities)
    }

    pub fn run_fabric_backend_probe(&self) -> FabricBackendSummary {
        let capabilities = self.discover_capabilities();
        let topology = run_fabric_topology_probe(&capabilities);
        run_fabric_backend_probe(&capabilities, &topology)
    }

    pub fn run_transport_capability_matrix_probe(
        &self,
    ) -> Result<TransportCapabilityMatrixSummary> {
        let capabilities = self.discover_capabilities();
        matrix_run::run_transport_capability_matrix_probe(self.config.device, &capabilities)
    }

    pub fn run_transport_registration_probe(&self) -> Result<TransportRegistrationSummary> {
        let _ = self.config;
        run_transport_registration_probe()
    }

    pub fn run_stage_pipeline_probe(
        &self,
        config: StagePipelineConfig,
    ) -> Result<StagePipelineSummary> {
        let capabilities = self.discover_capabilities();
        run::run_stage_pipeline_probe(config, self.config.device, &capabilities)
    }
}
