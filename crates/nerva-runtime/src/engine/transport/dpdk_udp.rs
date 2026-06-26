use nerva_core::types::error::Result;

use crate::engine::runtime::Runtime;
use crate::transport::dpdk_udp::config::DpdkUdpProbeConfig;
use crate::transport::dpdk_udp::run::run_dpdk_udp_protocol_probe;
use crate::transport::dpdk_udp::summary::DpdkUdpProtocolSummary;

impl Runtime {
    pub fn run_dpdk_udp_protocol_probe(
        &self,
        config: DpdkUdpProbeConfig,
    ) -> Result<DpdkUdpProtocolSummary> {
        let fabric = self.run_fabric_backend_probe();
        run_dpdk_udp_protocol_probe(config, fabric.dpdk_udp_gpu, fabric.dpdk_udp_pinned_host)
    }
}
