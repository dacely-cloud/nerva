use nerva_core::types::error::Result;

use crate::engine::runtime::Runtime;
use crate::transport::stage::config::StagePipelineConfig;
use crate::transport::stage::run;
use crate::transport::stage::summary::StagePipelineSummary;

impl Runtime {
    pub fn run_stage_pipeline_probe(
        &self,
        config: StagePipelineConfig,
    ) -> Result<StagePipelineSummary> {
        let capabilities = self.discover_capabilities();
        run::run_stage_pipeline_probe(config, self.config.device, &capabilities)
    }
}
