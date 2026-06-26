use crate::engine::runtime::Runtime;
use crate::transport::fabric::backend::probe::run_fabric_backend_probe;
use crate::transport::fabric::backend::types::FabricBackendSummary;
use crate::transport::fabric::probe::run_fabric_topology_probe;
use crate::transport::fabric::summary::FabricTopologySummary;

impl Runtime {
    pub fn run_fabric_topology_probe(&self) -> FabricTopologySummary {
        let capabilities = self.discover_capabilities();
        run_fabric_topology_probe(&capabilities)
    }

    pub fn run_fabric_backend_probe(&self) -> FabricBackendSummary {
        let capabilities = self.discover_capabilities();
        let topology = run_fabric_topology_probe(&capabilities);
        run_fabric_backend_probe(&capabilities, &topology)
    }
}
