use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;

use crate::common::dtype::dtype_to_str;
use crate::hf::architecture::HfArchitectureKind;
use crate::hf::contract::validate_weight_layout_contract;
use crate::hf::metadata::{HfMlpLayerKind, HfModelMetadata};
use crate::hf::validate::validate_hf_metadata;
use crate::weights::layout::entry::{WeightBlockRole, WeightBlockSpec};
use crate::weights::layout::plan::layer_blocks::{push_layer_weight_blocks, sum_weight_bytes};

mod layer_blocks;

#[derive(Clone, Debug, PartialEq)]
pub struct HfWeightLayoutPlan {
    pub metadata: HfModelMetadata,
    pub dtype: DType,
    pub blocks: Vec<WeightBlockSpec>,
    pub total_weight_bytes: usize,
    pub per_layer_weight_bytes: usize,
    pub static_weight_bytes: usize,
}

impl HfWeightLayoutPlan {
    pub fn to_json(&self) -> String {
        let moe_layers = self
            .metadata
            .mlp_layer_types
            .iter()
            .filter(|kind| **kind == HfMlpLayerKind::SparseMoe)
            .count();
        format!(
            "{{\"architecture\":\"{}\",\"dtype\":\"{}\",\"blocks\":{},\"layers\":{},\"moe_layers\":{},\"total_weight_bytes\":{},\"per_layer_weight_bytes\":{},\"static_weight_bytes\":{},\"hidden_size\":{},\"attention_hidden_size\":{},\"head_dim\":{},\"kv_hidden_size\":{},\"tie_word_embeddings\":{}}}",
            self.metadata.architecture.as_str(),
            dtype_to_str(self.dtype),
            self.blocks.len(),
            self.metadata.num_hidden_layers,
            moe_layers,
            self.total_weight_bytes,
            self.per_layer_weight_bytes,
            self.static_weight_bytes,
            self.metadata.hidden_size,
            self.metadata.attention_hidden(),
            self.metadata.head_dim(),
            self.metadata.kv_hidden(),
            self.metadata.tie_word_embeddings,
        )
    }
}

pub fn plan_hf_weight_layout(metadata: &HfModelMetadata) -> Result<HfWeightLayoutPlan> {
    validate_weight_layout_contract(metadata)?;
    if matches!(
        metadata.architecture,
        HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32
    ) {
        return plan_deepseek_v3_weight_layout(metadata);
    }
    metadata.block_shape().validate()?;
    validate_hf_metadata(
        metadata.hidden_size,
        metadata.num_hidden_layers,
        metadata.num_attention_heads,
        metadata.num_key_value_heads,
        metadata.head_dim,
        metadata.intermediate_size,
        metadata.vocab_size,
    )?;
    let dtype = metadata
        .torch_dtype
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "HF weight layout requires torch_dtype".to_string(),
        })?;
    let attention_hidden = metadata.attention_hidden();
    let kv_hidden = metadata.kv_hidden();

    let static_block_count = if metadata.tie_word_embeddings { 2 } else { 3 };
    let per_layer_blocks = if metadata.qk_norm { 11 } else { 9 };
    let mut blocks = Vec::with_capacity(
        metadata.num_hidden_layers.saturating_mul(per_layer_blocks) + static_block_count,
    );
    blocks.push(WeightBlockSpec::new(
        WeightBlockRole::TokenEmbedding,
        None,
        metadata.vocab_size,
        metadata.hidden_size,
        dtype,
        MemoryTier::Dram,
    )?);

    for layer in 0..metadata.num_hidden_layers {
        let layer = u32::try_from(layer).map_err(|_| NervaError::InvalidArgument {
            reason: "layer index does not fit u32".to_string(),
        })?;
        push_layer_weight_blocks(
            &mut blocks,
            metadata,
            attention_hidden,
            kv_hidden,
            dtype,
            layer,
        )?;
    }

    blocks.push(WeightBlockSpec::new(
        WeightBlockRole::FinalNorm,
        None,
        metadata.hidden_size,
        1,
        dtype,
        MemoryTier::Dram,
    )?);

    if !metadata.tie_word_embeddings {
        blocks.push(WeightBlockSpec::new(
            WeightBlockRole::LmHead,
            None,
            metadata.vocab_size,
            metadata.hidden_size,
            dtype,
            MemoryTier::Dram,
        )?);
    }

    let total_weight_bytes = sum_weight_bytes(&blocks)?;
    let static_weight_bytes = blocks
        .iter()
        .filter(|block| block.layer.is_none())
        .map(|block| block.bytes)
        .try_fold(0usize, |acc, bytes| {
            acc.checked_add(bytes)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes,
                    reason: "static weight byte count overflow".to_string(),
                })
        })?;
    let per_layer_weight_bytes = total_weight_bytes
        .checked_sub(static_weight_bytes)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: total_weight_bytes,
            reason: "weight byte accounting underflow".to_string(),
        })?
        / metadata.num_hidden_layers;

    Ok(HfWeightLayoutPlan {
        metadata: metadata.clone(),
        dtype,
        blocks,
        total_weight_bytes,
        per_layer_weight_bytes,
        static_weight_bytes,
    })
}

fn plan_deepseek_v3_weight_layout(metadata: &HfModelMetadata) -> Result<HfWeightLayoutPlan> {
    let dtype = metadata
        .torch_dtype
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "DeepSeek V3 weight layout requires torch_dtype".to_string(),
        })?;
    if dtype != DType::BF16 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek V3-family checkpoints are expected to declare bfloat16 torch_dtype"
                .to_string(),
        });
    }

    let q_lora_rank = required_metadata_usize(metadata.q_lora_rank, "q_lora_rank")?;
    let kv_lora_rank = required_metadata_usize(metadata.kv_lora_rank, "kv_lora_rank")?;
    let qk_nope_head_dim = required_metadata_usize(metadata.qk_nope_head_dim, "qk_nope_head_dim")?;
    let qk_rope_head_dim = required_metadata_usize(metadata.qk_rope_head_dim, "qk_rope_head_dim")?;
    let v_head_dim = required_metadata_usize(metadata.v_head_dim, "v_head_dim")?;
    let moe_intermediate =
        required_metadata_usize(metadata.moe_intermediate_size, "moe_intermediate_size")?;
    let shared_intermediate = metadata.shared_expert_intermediate_size.unwrap_or(0);
    let num_experts = required_metadata_usize(metadata.num_experts, "num_experts")?;

    let q_head_dim = qk_nope_head_dim
        .checked_add(qk_rope_head_dim)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: qk_nope_head_dim,
            reason: "DeepSeek V3 Q head dimension overflow".to_string(),
        })?;
    let q_rows = metadata
        .num_attention_heads
        .checked_mul(q_head_dim)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: q_head_dim,
            reason: "DeepSeek V3 q_b projection row count overflow".to_string(),
        })?;
    let kv_a_rows =
        kv_lora_rank
            .checked_add(qk_rope_head_dim)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: kv_lora_rank,
                reason: "DeepSeek V3 kv_a projection row count overflow".to_string(),
            })?;
    let kv_b_rows = metadata
        .num_attention_heads
        .checked_mul(qk_nope_head_dim.checked_add(v_head_dim).ok_or_else(|| {
            NervaError::AllocationFailed {
                bytes: qk_nope_head_dim,
                reason: "DeepSeek V3 kv_b head dimension overflow".to_string(),
            }
        })?)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: v_head_dim,
            reason: "DeepSeek V3 kv_b projection row count overflow".to_string(),
        })?;
    let value_hidden = metadata
        .num_attention_heads
        .checked_mul(v_head_dim)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: v_head_dim,
            reason: "DeepSeek V3 value hidden size overflow".to_string(),
        })?;

    let static_block_count = if metadata.tie_word_embeddings { 2 } else { 3 };
    let mut blocks = Vec::with_capacity(
        metadata
            .num_hidden_layers
            .saturating_mul(1600)
            .saturating_add(static_block_count),
    );
    blocks.push(WeightBlockSpec::new(
        WeightBlockRole::TokenEmbedding,
        None,
        metadata.vocab_size,
        metadata.hidden_size,
        DType::BF16,
        MemoryTier::Dram,
    )?);

    let norm_dtype = deepseek_v3_norm_dtype(metadata.architecture);
    for layer in 0..metadata.num_hidden_layers {
        let layer = u32::try_from(layer).map_err(|_| NervaError::InvalidArgument {
            reason: "layer index does not fit u32".to_string(),
        })?;
        push_deepseek_v3_layer_blocks(
            &mut blocks,
            metadata,
            layer,
            norm_dtype,
            q_lora_rank,
            kv_lora_rank,
            q_rows,
            kv_a_rows,
            kv_b_rows,
            value_hidden,
            moe_intermediate,
            shared_intermediate,
            num_experts,
        )?;
    }

    blocks.push(WeightBlockSpec::new(
        WeightBlockRole::FinalNorm,
        None,
        metadata.hidden_size,
        1,
        norm_dtype,
        MemoryTier::Dram,
    )?);
    if !metadata.tie_word_embeddings {
        blocks.push(WeightBlockSpec::new(
            WeightBlockRole::LmHead,
            None,
            metadata.vocab_size,
            metadata.hidden_size,
            DType::BF16,
            MemoryTier::Dram,
        )?);
    }

    let total_weight_bytes = sum_weight_bytes(&blocks)?;
    let static_weight_bytes = blocks
        .iter()
        .filter(|block| block.layer.is_none())
        .map(|block| block.bytes)
        .try_fold(0usize, |acc, bytes| {
            acc.checked_add(bytes)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes,
                    reason: "DeepSeek V3 static weight byte count overflow".to_string(),
                })
        })?;
    let per_layer_weight_bytes = total_weight_bytes
        .checked_sub(static_weight_bytes)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: total_weight_bytes,
            reason: "DeepSeek V3 weight byte accounting underflow".to_string(),
        })?
        / metadata.num_hidden_layers;

    Ok(HfWeightLayoutPlan {
        metadata: metadata.clone(),
        dtype,
        blocks,
        total_weight_bytes,
        per_layer_weight_bytes,
        static_weight_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn push_deepseek_v3_layer_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    layer: u32,
    norm_dtype: DType,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    q_rows: usize,
    kv_a_rows: usize,
    kv_b_rows: usize,
    value_hidden: usize,
    moe_intermediate: usize,
    shared_intermediate: usize,
    num_experts: usize,
) -> Result<()> {
    push_block(
        blocks,
        WeightBlockRole::AttentionNorm,
        layer,
        metadata.hidden_size,
        1,
        norm_dtype,
    )?;
    push_deepseek_v3_attention_blocks(
        blocks,
        metadata,
        layer,
        norm_dtype,
        q_lora_rank,
        kv_lora_rank,
        q_rows,
        kv_a_rows,
        kv_b_rows,
        value_hidden,
    )?;
    push_deepseek_v3_indexer_blocks(blocks, metadata, layer, q_lora_rank)?;
    push_block(
        blocks,
        WeightBlockRole::MlpNorm,
        layer,
        metadata.hidden_size,
        1,
        norm_dtype,
    )?;
    match metadata.mlp_layer_types.get(layer as usize).copied() {
        Some(HfMlpLayerKind::SparseMoe) => push_deepseek_v3_moe_blocks(
            blocks,
            metadata,
            layer,
            moe_intermediate,
            shared_intermediate,
            num_experts,
        ),
        _ => push_deepseek_v3_dense_mlp_blocks(blocks, metadata, layer),
    }
}

#[allow(clippy::too_many_arguments)]
fn push_deepseek_v3_attention_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    layer: u32,
    norm_dtype: DType,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    q_rows: usize,
    kv_a_rows: usize,
    kv_b_rows: usize,
    value_hidden: usize,
) -> Result<()> {
    push_block(
        blocks,
        WeightBlockRole::DeepSeekQALoraProjection,
        layer,
        q_lora_rank,
        metadata.hidden_size,
        DType::F8E4M3,
    )?;
    push_scale_block(
        blocks,
        WeightBlockRole::DeepSeekQALoraScaleInv,
        layer,
        q_lora_rank,
        metadata.hidden_size,
    )?;
    push_block(
        blocks,
        WeightBlockRole::DeepSeekQALoraNorm,
        layer,
        q_lora_rank,
        1,
        norm_dtype,
    )?;
    push_block(
        blocks,
        WeightBlockRole::DeepSeekQBProjection,
        layer,
        q_rows,
        q_lora_rank,
        DType::F8E4M3,
    )?;
    push_scale_block(
        blocks,
        WeightBlockRole::DeepSeekQBScaleInv,
        layer,
        q_rows,
        q_lora_rank,
    )?;
    push_block(
        blocks,
        WeightBlockRole::DeepSeekKvAProjection,
        layer,
        kv_a_rows,
        metadata.hidden_size,
        DType::F8E4M3,
    )?;
    push_scale_block(
        blocks,
        WeightBlockRole::DeepSeekKvAScaleInv,
        layer,
        kv_a_rows,
        metadata.hidden_size,
    )?;
    push_block(
        blocks,
        WeightBlockRole::DeepSeekKvANorm,
        layer,
        kv_lora_rank,
        1,
        norm_dtype,
    )?;
    push_block(
        blocks,
        WeightBlockRole::DeepSeekKvBProjection,
        layer,
        kv_b_rows,
        kv_lora_rank,
        DType::F8E4M3,
    )?;
    push_scale_block(
        blocks,
        WeightBlockRole::DeepSeekKvBScaleInv,
        layer,
        kv_b_rows,
        kv_lora_rank,
    )?;
    push_block(
        blocks,
        WeightBlockRole::OutputProjection,
        layer,
        metadata.hidden_size,
        value_hidden,
        DType::F8E4M3,
    )?;
    push_scale_block(
        blocks,
        WeightBlockRole::DeepSeekOutputScaleInv,
        layer,
        metadata.hidden_size,
        value_hidden,
    )
}

fn push_deepseek_v3_indexer_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    layer: u32,
    q_lora_rank: usize,
) -> Result<()> {
    if metadata.architecture != HfArchitectureKind::DeepSeekV32 {
        return Ok(());
    }
    let index_n_heads = required_metadata_usize(metadata.index_n_heads, "index_n_heads")?;
    let index_head_dim = required_metadata_usize(metadata.index_head_dim, "index_head_dim")?;
    let query_rows =
        index_n_heads
            .checked_mul(index_head_dim)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: index_head_dim,
                reason: "DeepSeek V3.2 indexer query row count overflow".to_string(),
            })?;
    push_block(
        blocks,
        WeightBlockRole::DeepSeekIndexerQueryProjection,
        layer,
        query_rows,
        q_lora_rank,
        DType::F8E4M3,
    )?;
    push_scale_block(
        blocks,
        WeightBlockRole::DeepSeekIndexerQueryScaleInv,
        layer,
        query_rows,
        q_lora_rank,
    )?;
    push_block(
        blocks,
        WeightBlockRole::DeepSeekIndexerKeyProjection,
        layer,
        index_head_dim,
        metadata.hidden_size,
        DType::F8E4M3,
    )?;
    push_scale_block(
        blocks,
        WeightBlockRole::DeepSeekIndexerKeyScaleInv,
        layer,
        index_head_dim,
        metadata.hidden_size,
    )?;
    push_block(
        blocks,
        WeightBlockRole::DeepSeekIndexerKeyNorm,
        layer,
        index_head_dim,
        1,
        DType::F32,
    )?;
    push_block(
        blocks,
        WeightBlockRole::DeepSeekIndexerKeyNormBias,
        layer,
        index_head_dim,
        1,
        DType::F32,
    )?;
    push_block(
        blocks,
        WeightBlockRole::DeepSeekIndexerWeightsProjection,
        layer,
        index_n_heads,
        metadata.hidden_size,
        DType::BF16,
    )
}

fn push_deepseek_v3_dense_mlp_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    layer: u32,
) -> Result<()> {
    for (role, scale_role, rows, cols) in [
        (
            WeightBlockRole::GateProjection,
            WeightBlockRole::GateScaleInv,
            metadata.intermediate_size,
            metadata.hidden_size,
        ),
        (
            WeightBlockRole::UpProjection,
            WeightBlockRole::UpScaleInv,
            metadata.intermediate_size,
            metadata.hidden_size,
        ),
        (
            WeightBlockRole::DownProjection,
            WeightBlockRole::DownScaleInv,
            metadata.hidden_size,
            metadata.intermediate_size,
        ),
    ] {
        push_block(blocks, role, layer, rows, cols, DType::F8E4M3)?;
        push_scale_block(blocks, scale_role, layer, rows, cols)?;
    }
    Ok(())
}

fn push_deepseek_v3_moe_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    layer: u32,
    moe_intermediate: usize,
    shared_intermediate: usize,
    num_experts: usize,
) -> Result<()> {
    push_block(
        blocks,
        WeightBlockRole::RouterProjection,
        layer,
        num_experts,
        metadata.hidden_size,
        DType::BF16,
    )?;
    if metadata.topk_method.as_deref() == Some("noaux_tc") {
        push_block(
            blocks,
            WeightBlockRole::RouterCorrectionBias,
            layer,
            num_experts,
            1,
            DType::F32,
        )?;
    }
    for (role, scale_role, rows, cols) in [
        (
            WeightBlockRole::SharedExpertGateProjection,
            WeightBlockRole::SharedExpertGateScaleInv,
            shared_intermediate,
            metadata.hidden_size,
        ),
        (
            WeightBlockRole::SharedExpertUpProjection,
            WeightBlockRole::SharedExpertUpScaleInv,
            shared_intermediate,
            metadata.hidden_size,
        ),
        (
            WeightBlockRole::SharedExpertDownProjection,
            WeightBlockRole::SharedExpertDownScaleInv,
            metadata.hidden_size,
            shared_intermediate,
        ),
    ] {
        if rows > 0 {
            push_block(blocks, role, layer, rows, cols, DType::F8E4M3)?;
            push_scale_block(blocks, scale_role, layer, rows, cols)?;
        }
    }
    for (role, scale_role, rows, cols) in [
        (
            WeightBlockRole::ExpertGateProjection,
            WeightBlockRole::ExpertGateScaleInv,
            moe_intermediate,
            metadata.hidden_size,
        ),
        (
            WeightBlockRole::ExpertUpProjection,
            WeightBlockRole::ExpertUpScaleInv,
            moe_intermediate,
            metadata.hidden_size,
        ),
        (
            WeightBlockRole::ExpertDownProjection,
            WeightBlockRole::ExpertDownScaleInv,
            metadata.hidden_size,
            moe_intermediate,
        ),
    ] {
        push_expert_block(blocks, role, layer, num_experts, rows, cols, DType::F8E4M3)?;
        push_expert_block(
            blocks,
            scale_role,
            layer,
            num_experts,
            scale_dim(rows),
            scale_dim(cols),
            DType::F32,
        )?;
    }
    Ok(())
}

fn push_block(
    blocks: &mut Vec<WeightBlockSpec>,
    role: WeightBlockRole,
    layer: u32,
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<()> {
    blocks.push(WeightBlockSpec::new(
        role,
        Some(layer),
        rows,
        cols,
        dtype,
        MemoryTier::Dram,
    )?);
    Ok(())
}

fn push_scale_block(
    blocks: &mut Vec<WeightBlockSpec>,
    role: WeightBlockRole,
    layer: u32,
    rows: usize,
    cols: usize,
) -> Result<()> {
    push_block(
        blocks,
        role,
        layer,
        scale_dim(rows),
        scale_dim(cols),
        DType::F32,
    )
}

fn push_expert_block(
    blocks: &mut Vec<WeightBlockSpec>,
    role: WeightBlockRole,
    layer: u32,
    depth: usize,
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<()> {
    blocks.push(WeightBlockSpec::new_rank3(
        role,
        Some(layer),
        depth,
        rows,
        cols,
        dtype,
        MemoryTier::Dram,
    )?);
    Ok(())
}

fn scale_dim(value: usize) -> usize {
    value.div_ceil(128)
}

fn deepseek_v3_norm_dtype(architecture: HfArchitectureKind) -> DType {
    if architecture == HfArchitectureKind::DeepSeekV32 {
        DType::F32
    } else {
        DType::BF16
    }
}

fn required_metadata_usize(value: Option<usize>, key: &'static str) -> Result<usize> {
    value.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("DeepSeek V3 metadata is missing {key}"),
    })
}
