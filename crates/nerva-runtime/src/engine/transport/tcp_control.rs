use nerva_core::types::error::Result;

use crate::engine::runtime::Runtime;
use crate::transport::tcp_control::config::TcpControlProbeConfig;
use crate::transport::tcp_control::run::run_tcp_control_probe;
use crate::transport::tcp_control::summary::TcpControlSummary;

impl Runtime {
    pub fn run_tcp_control_probe(
        &self,
        config: TcpControlProbeConfig,
    ) -> Result<TcpControlSummary> {
        let _ = self.config;
        run_tcp_control_probe(config)
    }
}
