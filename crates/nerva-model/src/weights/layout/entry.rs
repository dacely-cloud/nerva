use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;

use crate::common::dtype::dtype_size_bytes;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WeightBlockRole {
    TokenEmbedding,
    AttentionNorm,
    QueryProjection,
    QueryNorm,
    QueryBias,
    KeyProjection,
    KeyNorm,
    KeyBias,
    ValueProjection,
    ValueBias,
    OutputProjection,
    OutputBias,
    LinearConvProjection,
    LinearQkvProjection,
    LinearZProjection,
    LinearBProjection,
    LinearAProjection,
    LinearDtBias,
    LinearALog,
    LinearNorm,
    LinearOutputProjection,
    MlpNorm,
    GateProjection,
    UpProjection,
    DownProjection,
    RouterProjection,
    ExpertGateProjection,
    ExpertUpProjection,
    ExpertGateUpProjection,
    ExpertDownProjection,
    SharedExpertGateProjection,
    SharedExpertUpProjection,
    SharedExpertDownProjection,
    SharedExpertRouterProjection,
    FinalNorm,
    LmHead,
}

impl WeightBlockRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TokenEmbedding => "token_embedding",
            Self::AttentionNorm => "attention_norm",
            Self::QueryProjection => "q_proj",
            Self::QueryNorm => "q_norm",
            Self::QueryBias => "q_proj_bias",
            Self::KeyProjection => "k_proj",
            Self::KeyNorm => "k_norm",
            Self::KeyBias => "k_proj_bias",
            Self::ValueProjection => "v_proj",
            Self::ValueBias => "v_proj_bias",
            Self::OutputProjection => "o_proj",
            Self::OutputBias => "o_proj_bias",
            Self::LinearConvProjection => "linear_conv1d",
            Self::LinearQkvProjection => "linear_in_proj_qkv",
            Self::LinearZProjection => "linear_in_proj_z",
            Self::LinearBProjection => "linear_in_proj_b",
            Self::LinearAProjection => "linear_in_proj_a",
            Self::LinearDtBias => "linear_dt_bias",
            Self::LinearALog => "linear_a_log",
            Self::LinearNorm => "linear_norm",
            Self::LinearOutputProjection => "linear_out_proj",
            Self::MlpNorm => "mlp_norm",
            Self::GateProjection => "gate_proj",
            Self::UpProjection => "up_proj",
            Self::DownProjection => "down_proj",
            Self::RouterProjection => "router",
            Self::ExpertGateProjection => "expert_gate_proj",
            Self::ExpertUpProjection => "expert_up_proj",
            Self::ExpertGateUpProjection => "expert_gate_up_proj",
            Self::ExpertDownProjection => "expert_down_proj",
            Self::SharedExpertGateProjection => "shared_expert_gate_proj",
            Self::SharedExpertUpProjection => "shared_expert_up_proj",
            Self::SharedExpertDownProjection => "shared_expert_down_proj",
            Self::SharedExpertRouterProjection => "shared_expert_gate",
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
    pub depth: Option<usize>,
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
        Self::new_with_depth(role, layer, None, rows, cols, dtype, tier)
    }

    pub(crate) fn new_rank3(
        role: WeightBlockRole,
        layer: Option<u32>,
        depth: usize,
        rows: usize,
        cols: usize,
        dtype: DType,
        tier: MemoryTier,
    ) -> Result<Self> {
        if depth == 0 {
            return Err(NervaError::InvalidArgument {
                reason: format!("weight block {} depth must be non-zero", role.as_str()),
            });
        }
        Self::new_with_depth(role, layer, Some(depth), rows, cols, dtype, tier)
    }

    fn new_with_depth(
        role: WeightBlockRole,
        layer: Option<u32>,
        depth: Option<usize>,
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
            .and_then(|elements| match depth {
                Some(depth) => elements.checked_mul(depth),
                None => Some(elements),
            })
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
            depth,
            elements,
            bytes,
            dtype,
            tier,
        })
    }
}
