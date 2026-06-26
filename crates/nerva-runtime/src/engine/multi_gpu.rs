use nerva_core::types::error::Result;

use crate::engine::runtime::Runtime;
use crate::mgpu::config::MultiGpuNodeConfig;
use crate::mgpu::run::run_multi_gpu_node_probe;
use crate::mgpu::summary::MultiGpuNodeSummary;

impl Runtime {
    pub fn run_multi_gpu_node_probe(
        &self,
        config: MultiGpuNodeConfig,
    ) -> Result<MultiGpuNodeSummary> {
        run_multi_gpu_node_probe(config)
    }
}
