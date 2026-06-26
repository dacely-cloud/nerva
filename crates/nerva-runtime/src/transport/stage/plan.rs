use crate::capabilities::snapshot::CapabilitySnapshot;
use crate::transport::path::{TransportPathDecision, TransportPathRequest, plan_transport_path};
use crate::transport::stage::config::StagePipelineConfig;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::DeviceOrdinal;
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StageSpec {
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_count: u32,
    pub weight_bytes: usize,
    pub kv_bytes: usize,
    pub owner: ExecutionOwner,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StageBoundaryPlan {
    pub boundary_index: u32,
    pub source_stage: u32,
    pub destination_stage: u32,
    pub activation_bytes: usize,
    pub moved_weight_bytes: usize,
    pub all_reduce_bytes: usize,
    pub decision: TransportPathDecision,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagePipelinePlan {
    pub config: StagePipelineConfig,
    pub stages: Vec<StageSpec>,
    pub boundaries: Vec<StageBoundaryPlan>,
}

pub fn plan_stage_pipeline(
    config: StagePipelineConfig,
    device: DeviceOrdinal,
    capabilities: &CapabilitySnapshot,
) -> Result<StagePipelinePlan> {
    validate_stage_config(config)?;
    let stages = build_stages(config, device)?;
    let activation_bytes = config.activation_bytes()?;
    let mut boundaries = Vec::with_capacity(config.stages.saturating_sub(1) as usize);

    for boundary_index in 0..config.stages - 1 {
        let request = TransportPathRequest::from_capabilities(
            MemoryTier::Vram,
            MemoryTier::Vram,
            activation_bytes,
            config.mode,
            ExecutionOwner::Gpu(device),
            capabilities,
        );
        let decision = plan_transport_path(request)?;
        boundaries.push(StageBoundaryPlan {
            boundary_index,
            source_stage: boundary_index,
            destination_stage: boundary_index + 1,
            activation_bytes,
            moved_weight_bytes: 0,
            all_reduce_bytes: 0,
            decision,
        });
    }

    Ok(StagePipelinePlan {
        config,
        stages,
        boundaries,
    })
}

fn validate_stage_config(config: StagePipelineConfig) -> Result<()> {
    if config.stages < 2 {
        return Err(NervaError::InvalidArgument {
            reason: "stage pipeline requires at least two stages".to_string(),
        });
    }
    if config.hidden_size == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "stage pipeline hidden size must be non-zero".to_string(),
        });
    }
    if config.layers_per_stage == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "stage pipeline layers per stage must be non-zero".to_string(),
        });
    }
    if config.weight_bytes_per_stage == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "stage pipeline weight bytes must be non-zero".to_string(),
        });
    }
    if config.kv_bytes_per_stage == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "stage pipeline KV bytes must be non-zero".to_string(),
        });
    }
    let _ = config.activation_bytes()?;
    Ok(())
}

fn build_stages(config: StagePipelineConfig, device: DeviceOrdinal) -> Result<Vec<StageSpec>> {
    let mut stages = Vec::with_capacity(config.stages as usize);
    for stage_index in 0..config.stages {
        stages.push(StageSpec {
            stage_index,
            layer_start: stage_index
                .checked_mul(config.layers_per_stage)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: "stage layer range overflowed".to_string(),
                })?,
            layer_count: config.layers_per_stage,
            weight_bytes: config.weight_bytes_per_stage,
            kv_bytes: config.kv_bytes_per_stage,
            owner: ExecutionOwner::Gpu(device),
        });
    }
    Ok(stages)
}
