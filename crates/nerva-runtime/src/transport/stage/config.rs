use crate::transport::path::types::TransferMode;
use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StagePipelineConfig {
    pub stages: u32,
    pub hidden_size: usize,
    pub dtype: DType,
    pub layers_per_stage: u32,
    pub weight_bytes_per_stage: usize,
    pub kv_bytes_per_stage: usize,
    pub mode: TransferMode,
}

impl StagePipelineConfig {
    pub const fn reference_decode() -> Self {
        Self {
            stages: 4,
            hidden_size: 16_384,
            dtype: DType::F16,
            layers_per_stage: 20,
            weight_bytes_per_stage: 200 * 1024 * 1024 * 1024,
            kv_bytes_per_stage: 4 * 1024 * 1024,
            mode: TransferMode::Decode,
        }
    }

    pub fn activation_bytes(self) -> Result<usize> {
        self.hidden_size
            .checked_mul(dtype_bytes(self.dtype)?)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "stage activation byte size overflowed".to_string(),
            })
    }
}

fn dtype_bytes(dtype: DType) -> Result<usize> {
    match dtype {
        DType::F16 | DType::BF16 | DType::U16 => Ok(2),
        DType::F32 | DType::U32 | DType::I32 => Ok(4),
        DType::U8 => Ok(1),
    }
}
