use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;

use crate::common::dtype::dtype_size_bytes;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WeightBlockRole {
    TokenEmbedding,
    AttentionNorm,
    QueryProjection,
    KeyProjection,
    ValueProjection,
    OutputProjection,
    MlpNorm,
    GateProjection,
    UpProjection,
    DownProjection,
    FinalNorm,
    LmHead,
}

impl WeightBlockRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TokenEmbedding => "token_embedding",
            Self::AttentionNorm => "attention_norm",
            Self::QueryProjection => "q_proj",
            Self::KeyProjection => "k_proj",
            Self::ValueProjection => "v_proj",
            Self::OutputProjection => "o_proj",
            Self::MlpNorm => "mlp_norm",
            Self::GateProjection => "gate_proj",
            Self::UpProjection => "up_proj",
            Self::DownProjection => "down_proj",
            Self::FinalNorm => "final_norm",
            Self::LmHead => "lm_head",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct WeightBlockSpec {
    pub role: WeightBlockRole,
    pub layer: Option<u32>,
    pub rows: usize,
    pub cols: usize,
    pub elements: usize,
    pub bytes: usize,
    pub dtype: DType,
    pub tier: MemoryTier,
}

impl WeightBlockSpec {
    pub(crate) fn new(
        role: WeightBlockRole,
        layer: Option<u32>,
        rows: usize,
        cols: usize,
        dtype: DType,
        tier: MemoryTier,
    ) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(NervaError::InvalidArgument {
                reason: format!("weight block {} shape must be non-zero", role.as_str()),
            });
        }
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: 0,
                reason: format!("weight block {} element count overflow", role.as_str()),
            })?;
        let bytes = elements
            .checked_mul(dtype_size_bytes(dtype)?)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: elements,
                reason: format!("weight block {} byte count overflow", role.as_str()),
            })?;
        Ok(Self {
            role,
            layer,
            rows,
            cols,
            elements,
            bytes,
            dtype,
            tier,
        })
    }
}
