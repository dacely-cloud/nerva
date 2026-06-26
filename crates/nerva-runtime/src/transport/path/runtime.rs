use nerva_core::types::error::Result;

use crate::engine::runtime::Runtime;
use crate::transport::path::decision::TransportPathDecision;
use crate::transport::path::planner;
use crate::transport::path::request::TransportPathRequest;
use crate::transport::probe::run as transport_probe_run;
use crate::transport::probe::summary::TransportPathProbeSummary;

impl Runtime {
    pub fn plan_transport_path(
        &self,
        request: TransportPathRequest,
    ) -> Result<TransportPathDecision> {
        let _ = self.config;
        planner::plan_transport_path(request)
    }

    pub fn run_transport_path_probe(&self) -> Result<TransportPathProbeSummary> {
        let capabilities = self.discover_capabilities();
        transport_probe_run::run_transport_path_probe(self.config.device, &capabilities)
    }
}
