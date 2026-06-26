use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::ownership::owner::ExecutionOwner;

use crate::mgpu::config::{MultiGpuNodeConfig, validate_multi_gpu_config};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum GpuIslandRole {
    Compute,
    EgressCompute,
}

impl GpuIslandRole {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::EgressCompute => "egress-compute",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GpuIslandPlan {
    pub gpu: DeviceOrdinal,
    pub role: GpuIslandRole,
    pub layer_start: u32,
    pub layer_count: u32,
    pub local_vram_bytes: usize,
    pub hot_weight_bytes: usize,
    pub kv_bytes: usize,
    pub dram_weight_backing_bytes: usize,
    pub max_single_allocation_bytes: usize,
    pub owner: ExecutionOwner,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct InterGpuBoundaryPlan {
    pub boundary_index: u32,
    pub source_gpu: DeviceOrdinal,
    pub destination_gpu: DeviceOrdinal,
    pub activation_bytes: usize,
    pub moved_weight_bytes: usize,
    pub all_reduce_bytes: usize,
    pub phase_handoff_required: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiGpuNodePlan {
    pub config: MultiGpuNodeConfig,
    pub islands: Vec<GpuIslandPlan>,
    pub boundaries: Vec<InterGpuBoundaryPlan>,
    pub aggregate_vram_pool_claimed: bool,
    pub coherent_vram_allocation_claims: u32,
}

pub fn plan_multi_gpu_node(config: MultiGpuNodeConfig) -> Result<MultiGpuNodePlan> {
    validate_multi_gpu_config(config)?;
    let islands = build_islands(config)?;
    let boundaries = build_boundaries(config)?;

    Ok(MultiGpuNodePlan {
        config,
        islands,
        boundaries,
        aggregate_vram_pool_claimed: false,
        coherent_vram_allocation_claims: 0,
    })
}

fn build_islands(config: MultiGpuNodeConfig) -> Result<Vec<GpuIslandPlan>> {
    let mut islands = Vec::with_capacity(config.gpu_count as usize);
    let total_hot_cache = config
        .hot_weight_cache_bytes_per_gpu
        .checked_mul(config.gpu_count as usize)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "multi-GPU hot cache byte count overflowed".to_string(),
        })?;
    let total_dram_backing = config.stage_weight_bytes.saturating_sub(total_hot_cache);
    let base_layers = config.layers / config.gpu_count;
    let remainder_layers = config.layers % config.gpu_count;

    for gpu_index in 0..config.gpu_count {
        let extra = u32::from(gpu_index < remainder_layers);
        let layer_count = base_layers + extra;
        let layer_start = layer_start_for_gpu(config, gpu_index);
        let hot_weight_bytes = config
            .hot_weight_cache_bytes_per_gpu
            .min(config.stage_weight_bytes);
        let dram_weight_backing_bytes =
            split_bytes_across_gpus(total_dram_backing, config.gpu_count, gpu_index)?;
        let kv_bytes = config.kv_cache_bytes_per_gpu.min(config.stage_kv_bytes);
        let max_single_allocation_bytes = hot_weight_bytes.max(kv_bytes);
        islands.push(GpuIslandPlan {
            gpu: DeviceOrdinal(gpu_index as i32),
            role: if gpu_index == config.nic_near_gpu {
                GpuIslandRole::EgressCompute
            } else {
                GpuIslandRole::Compute
            },
            layer_start,
            layer_count,
            local_vram_bytes: config.local_vram_bytes_per_gpu,
            hot_weight_bytes,
            kv_bytes,
            dram_weight_backing_bytes,
            max_single_allocation_bytes,
            owner: ExecutionOwner::Gpu(DeviceOrdinal(gpu_index as i32)),
        });
    }
    Ok(islands)
}

fn build_boundaries(config: MultiGpuNodeConfig) -> Result<Vec<InterGpuBoundaryPlan>> {
    let mut boundaries = Vec::with_capacity(config.gpu_count.saturating_sub(1) as usize);
    let activation_bytes = config.activation_bytes()?;
    for boundary_index in 0..config.gpu_count - 1 {
        boundaries.push(InterGpuBoundaryPlan {
            boundary_index,
            source_gpu: DeviceOrdinal(boundary_index as i32),
            destination_gpu: DeviceOrdinal((boundary_index + 1) as i32),
            activation_bytes,
            moved_weight_bytes: 0,
            all_reduce_bytes: 0,
            phase_handoff_required: true,
        });
    }
    Ok(boundaries)
}

fn layer_start_for_gpu(config: MultiGpuNodeConfig, gpu_index: u32) -> u32 {
    let base_layers = config.layers / config.gpu_count;
    let remainder_layers = config.layers % config.gpu_count;
    gpu_index * base_layers + gpu_index.min(remainder_layers)
}

fn split_bytes_across_gpus(total: usize, gpu_count: u32, gpu_index: u32) -> Result<usize> {
    if total == 0 {
        return Ok(0);
    }
    let gpu_count = gpu_count as usize;
    let base = total / gpu_count;
    let remainder = total % gpu_count;
    base.checked_add(usize::from((gpu_index as usize) < remainder))
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "multi-GPU byte split overflowed".to_string(),
        })
}
