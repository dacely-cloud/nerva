use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MultiGpuNodeConfig {
    pub gpu_count: u32,
    pub local_vram_bytes_per_gpu: usize,
    pub hidden_size: usize,
    pub dtype: DType,
    pub layers: u32,
    pub stage_weight_bytes: usize,
    pub stage_kv_bytes: usize,
    pub hot_weight_cache_bytes_per_gpu: usize,
    pub kv_cache_bytes_per_gpu: usize,
    pub nic_near_gpu: u32,
}

impl MultiGpuNodeConfig {
    pub const fn reference_2080ti_stage() -> Self {
        Self {
            gpu_count: 8,
            local_vram_bytes_per_gpu: 11 * 1024 * 1024 * 1024,
            hidden_size: 16_384,
            dtype: DType::F16,
            layers: 20,
            stage_weight_bytes: 200 * 1024 * 1024 * 1024,
            stage_kv_bytes: 4 * 1024 * 1024,
            hot_weight_cache_bytes_per_gpu: 8 * 1024 * 1024 * 1024,
            kv_cache_bytes_per_gpu: 512 * 1024,
            nic_near_gpu: 7,
        }
    }

    pub fn activation_bytes(self) -> Result<usize> {
        self.hidden_size
            .checked_mul(dtype_bytes(self.dtype)?)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "multi-GPU activation byte size overflowed".to_string(),
            })
    }

    pub fn aggregate_vram_bytes(self) -> Result<usize> {
        self.local_vram_bytes_per_gpu
            .checked_mul(self.gpu_count as usize)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "multi-GPU aggregate VRAM byte count overflowed".to_string(),
            })
    }
}

pub(crate) fn validate_multi_gpu_config(config: MultiGpuNodeConfig) -> Result<()> {
    if config.gpu_count < 2 {
        return Err(NervaError::InvalidArgument {
            reason: "same-node multi-GPU planning requires at least two GPUs".to_string(),
        });
    }
    if config.local_vram_bytes_per_gpu == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "multi-GPU local VRAM per GPU must be non-zero".to_string(),
        });
    }
    if config.hidden_size == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "multi-GPU hidden size must be non-zero".to_string(),
        });
    }
    if config.layers == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "multi-GPU layer count must be non-zero".to_string(),
        });
    }
    if config.stage_weight_bytes == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "multi-GPU stage weight bytes must be non-zero".to_string(),
        });
    }
    if config.stage_kv_bytes == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "multi-GPU stage KV bytes must be non-zero".to_string(),
        });
    }
    if config.hot_weight_cache_bytes_per_gpu == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "multi-GPU hot weight cache per GPU must be non-zero".to_string(),
        });
    }
    if config.hot_weight_cache_bytes_per_gpu > config.local_vram_bytes_per_gpu {
        return Err(NervaError::InvalidArgument {
            reason: "multi-GPU hot cache cannot exceed local GPU VRAM".to_string(),
        });
    }
    if config.kv_cache_bytes_per_gpu > config.local_vram_bytes_per_gpu {
        return Err(NervaError::InvalidArgument {
            reason: "multi-GPU KV cache cannot exceed local GPU VRAM".to_string(),
        });
    }
    if config.nic_near_gpu >= config.gpu_count {
        return Err(NervaError::InvalidArgument {
            reason: "multi-GPU NIC-near GPU index is outside the node".to_string(),
        });
    }
    let _ = config.activation_bytes()?;
    let _ = config.aggregate_vram_bytes()?;
    Ok(())
}

fn dtype_bytes(dtype: DType) -> Result<usize> {
    match dtype {
        DType::F16 | DType::BF16 | DType::U16 => Ok(2),
        DType::F32 | DType::U32 | DType::I32 => Ok(4),
        DType::I64 => Ok(8),
        DType::U8 | DType::I8 | DType::F8E4M3 | DType::F8E8M0 => Ok(1),
    }
}
