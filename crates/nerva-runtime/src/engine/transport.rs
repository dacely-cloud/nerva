use nerva_core::types::error::Result;

use crate::engine::runtime::Runtime;
use crate::transport::dpdk_udp::config::DpdkUdpProbeConfig;
use crate::transport::dpdk_udp::run::run_dpdk_udp_protocol_probe;
use crate::transport::dpdk_udp::summary::DpdkUdpProtocolSummary;
use crate::transport::fabric::backend::probe::run_fabric_backend_probe;
use crate::transport::fabric::backend::types::FabricBackendSummary;
use crate::transport::fabric::probe::run_fabric_topology_probe;
use crate::transport::fabric::summary::FabricTopologySummary;
use crate::transport::matrix::run as matrix_run;
use crate::transport::matrix::types::TransportCapabilityMatrixSummary;
use crate::transport::path::decision::TransportPathDecision;
use crate::transport::path::planner;
use crate::transport::path::request::TransportPathRequest;
use crate::transport::probe::run as transport_probe_run;
use crate::transport::probe::summary::TransportPathProbeSummary;
use crate::transport::registration::lifetime::run::run_transport_registration_lifecycle_probe;
use crate::transport::registration::lifetime::summary::TransportRegistrationLifecycleSummary;
use crate::transport::registration::probe::run::run_transport_registration_probe;
use crate::transport::registration::summary::TransportRegistrationSummary;
use crate::transport::stage::config::StagePipelineConfig;
use crate::transport::stage::run;
use crate::transport::stage::summary::StagePipelineSummary;
use crate::transport::tcp_control::config::TcpControlProbeConfig;
use crate::transport::tcp_control::run::run_tcp_control_probe;
use crate::transport::tcp_control::summary::TcpControlSummary;

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

    pub fn run_fabric_topology_probe(&self) -> FabricTopologySummary {
        let capabilities = self.discover_capabilities();
        run_fabric_topology_probe(&capabilities)
    }

    pub fn run_fabric_backend_probe(&self) -> FabricBackendSummary {
        let capabilities = self.discover_capabilities();
        let topology = run_fabric_topology_probe(&capabilities);
        run_fabric_backend_probe(&capabilities, &topology)
    }

    pub fn run_dpdk_udp_protocol_probe(
        &self,
        config: DpdkUdpProbeConfig,
    ) -> Result<DpdkUdpProtocolSummary> {
        let fabric = self.run_fabric_backend_probe();
        run_dpdk_udp_protocol_probe(config, fabric.dpdk_udp_gpu, fabric.dpdk_udp_pinned_host)
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

    pub fn run_transport_registration_lifecycle_probe(
        &self,
    ) -> Result<TransportRegistrationLifecycleSummary> {
        let _ = self.config;
        run_transport_registration_lifecycle_probe()
    }

    pub fn run_stage_pipeline_probe(
        &self,
        config: StagePipelineConfig,
    ) -> Result<StagePipelineSummary> {
        let capabilities = self.discover_capabilities();
        run::run_stage_pipeline_probe(config, self.config.device, &capabilities)
    }

    pub fn run_tcp_control_probe(
        &self,
        config: TcpControlProbeConfig,
    ) -> Result<TcpControlSummary> {
        let _ = self.config;
        run_tcp_control_probe(config)
    }
}
