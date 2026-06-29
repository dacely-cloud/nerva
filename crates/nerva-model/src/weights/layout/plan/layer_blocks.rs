use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;

use crate::hf::architecture::HfArchitectureKind;
use crate::hf::metadata::{HfAttentionLayerKind, HfMlpLayerKind, HfModelMetadata};
use crate::weights::layout::entry::{WeightBlockRole, WeightBlockSpec};

pub(super) fn push_layer_weight_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    attention_hidden: usize,
    kv_hidden: usize,
    dtype: DType,
    layer: u32,
) -> Result<()> {
    let hidden = metadata.hidden_size;
    push_layer_block(
        blocks,
        WeightBlockRole::AttentionNorm,
        hidden,
        1,
        dtype,
        layer,
    )?;
    if metadata
        .attention_layer_types
        .get(layer as usize)
        .is_some_and(|kind| *kind == HfAttentionLayerKind::Linear)
    {
        push_linear_attention_blocks(blocks, metadata, dtype, layer)?;
        push_mlp_norm_and_mlp_blocks(blocks, metadata, dtype, layer)?;
        return Ok(());
    }

    let query_rows = query_projection_rows(metadata, attention_hidden)?;
    push_layer_block(
        blocks,
        WeightBlockRole::QueryProjection,
        query_rows,
        hidden,
        dtype,
        layer,
    )?;
    push_qk_blocks(blocks, metadata, kv_hidden, dtype, layer)?;
    push_full_attention_tail_blocks(blocks, metadata, attention_hidden, kv_hidden, dtype, layer)?;
    push_mlp_norm_and_mlp_blocks(blocks, metadata, dtype, layer)?;
    push_attention_bias_blocks(blocks, metadata, query_rows, kv_hidden, dtype, layer)
}

fn push_linear_attention_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    dtype: DType,
    layer: u32,
) -> Result<()> {
    let dims = linear_attention_dims(metadata)?;
    for (role, rows, cols, block_dtype) in [
        (
            WeightBlockRole::LinearConvProjection,
            dims.conv_rows,
            dims.conv_kernel,
            dtype,
        ),
        (
            WeightBlockRole::LinearQkvProjection,
            dims.key_dim
                .checked_mul(2)
                .and_then(|rows| rows.checked_add(dims.value_dim))
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes: dims.key_dim,
                    reason: "Qwen3.5 GDN qkv projection row count overflow".to_string(),
                })?,
            metadata.hidden_size,
            dtype,
        ),
        (
            WeightBlockRole::LinearZProjection,
            dims.value_dim,
            metadata.hidden_size,
            dtype,
        ),
        (
            WeightBlockRole::LinearBProjection,
            dims.value_heads,
            metadata.hidden_size,
            dtype,
        ),
        (
            WeightBlockRole::LinearAProjection,
            dims.value_heads,
            metadata.hidden_size,
            dtype,
        ),
        (WeightBlockRole::LinearDtBias, dims.value_heads, 1, dtype),
        (WeightBlockRole::LinearALog, dims.value_heads, 1, DType::F32),
        (
            WeightBlockRole::LinearNorm,
            dims.value_head_dim,
            1,
            DType::F32,
        ),
        (
            WeightBlockRole::LinearOutputProjection,
            metadata.hidden_size,
            dims.value_dim,
            dtype,
        ),
    ] {
        push_layer_block(blocks, role, rows, cols, block_dtype, layer)?;
    }
    Ok(())
}

struct LinearAttentionDims {
    value_heads: usize,
    value_head_dim: usize,
    conv_kernel: usize,
    key_dim: usize,
    value_dim: usize,
    conv_rows: usize,
}

fn linear_attention_dims(metadata: &HfModelMetadata) -> Result<LinearAttentionDims> {
    let key_heads = metadata
        .linear_num_key_heads
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "Qwen3.5 linear_attention is missing linear_num_key_heads".to_string(),
        })?;
    let value_heads =
        metadata
            .linear_num_value_heads
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "Qwen3.5 linear_attention is missing linear_num_value_heads".to_string(),
            })?;
    let key_head_dim = metadata
        .linear_key_head_dim
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "Qwen3.5 linear_attention is missing linear_key_head_dim".to_string(),
        })?;
    let value_head_dim =
        metadata
            .linear_value_head_dim
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "Qwen3.5 linear_attention is missing linear_value_head_dim".to_string(),
            })?;
    let conv_kernel =
        metadata
            .linear_conv_kernel_dim
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "Qwen3.5 linear_attention is missing linear_conv_kernel_dim".to_string(),
            })?;
    let key_dim =
        key_heads
            .checked_mul(key_head_dim)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: key_heads,
                reason: "Qwen3.5 GDN key dimension overflow".to_string(),
            })?;
    let value_dim =
        value_heads
            .checked_mul(value_head_dim)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: value_heads,
                reason: "Qwen3.5 GDN value dimension overflow".to_string(),
            })?;
    let conv_rows = key_dim
        .checked_mul(2)
        .and_then(|rows| rows.checked_add(value_dim))
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: key_dim,
            reason: "Qwen3.5 GDN conv row count overflow".to_string(),
        })?;
    Ok(LinearAttentionDims {
        value_heads,
        value_head_dim,
        conv_kernel,
        key_dim,
        value_dim,
        conv_rows,
    })
}

fn query_projection_rows(metadata: &HfModelMetadata, attention_hidden: usize) -> Result<usize> {
    if matches!(
        metadata.architecture,
        HfArchitectureKind::Qwen35 | HfArchitectureKind::Qwen35Moe
    ) {
        attention_hidden
            .checked_mul(2)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: attention_hidden,
                reason: "Qwen3.5 query/gate projection row count overflow".to_string(),
            })
    } else {
        Ok(attention_hidden)
    }
}

fn push_qk_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    kv_hidden: usize,
    dtype: DType,
    layer: u32,
) -> Result<()> {
    if metadata.qk_norm {
        push_layer_block(
            blocks,
            WeightBlockRole::QueryNorm,
            metadata.head_dim,
            1,
            dtype,
            layer,
        )?;
    }
    push_layer_block(
        blocks,
        WeightBlockRole::KeyProjection,
        kv_hidden,
        metadata.hidden_size,
        dtype,
        layer,
    )?;
    if metadata.qk_norm {
        push_layer_block(
            blocks,
            WeightBlockRole::KeyNorm,
            metadata.head_dim,
            1,
            dtype,
            layer,
        )?;
    }
    Ok(())
}

fn push_full_attention_tail_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    attention_hidden: usize,
    kv_hidden: usize,
    dtype: DType,
    layer: u32,
) -> Result<()> {
    for (role, rows, cols) in [
        (
            WeightBlockRole::ValueProjection,
            kv_hidden,
            metadata.hidden_size,
        ),
        (
            WeightBlockRole::OutputProjection,
            metadata.hidden_size,
            attention_hidden,
        ),
    ] {
        push_layer_block(blocks, role, rows, cols, dtype, layer)?;
    }
    Ok(())
}

fn push_mlp_norm_and_mlp_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    dtype: DType,
    layer: u32,
) -> Result<()> {
    push_layer_block(
        blocks,
        WeightBlockRole::MlpNorm,
        metadata.hidden_size,
        1,
        dtype,
        layer,
    )?;
    match metadata.mlp_layer_types.get(layer as usize).copied() {
        Some(HfMlpLayerKind::SparseMoe) => push_sparse_moe_blocks(blocks, metadata, dtype, layer),
        _ => push_dense_mlp_projection_blocks(blocks, metadata, dtype, layer),
    }
}

fn push_dense_mlp_projection_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    dtype: DType,
    layer: u32,
) -> Result<()> {
    for (role, rows, cols) in [
        (
            WeightBlockRole::GateProjection,
            metadata.intermediate_size,
            metadata.hidden_size,
        ),
        (
            WeightBlockRole::UpProjection,
            metadata.intermediate_size,
            metadata.hidden_size,
        ),
        (
            WeightBlockRole::DownProjection,
            metadata.hidden_size,
            metadata.intermediate_size,
        ),
    ] {
        push_layer_block(blocks, role, rows, cols, dtype, layer)?;
    }
    Ok(())
}

fn push_sparse_moe_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    dtype: DType,
    layer: u32,
) -> Result<()> {
    let num_experts = metadata
        .num_experts
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "HF MoE layer is missing num_experts".to_string(),
        })?;
    let moe_intermediate =
        metadata
            .moe_intermediate_size
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "HF MoE layer is missing moe_intermediate_size".to_string(),
            })?;
    push_layer_block(
        blocks,
        WeightBlockRole::RouterProjection,
        num_experts,
        metadata.hidden_size,
        dtype,
        layer,
    )?;
    push_layer_block_rank3(
        blocks,
        WeightBlockRole::ExpertGateUpProjection,
        num_experts,
        moe_intermediate
            .checked_mul(2)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: moe_intermediate,
                reason: "HF MoE gate/up row count overflow".to_string(),
            })?,
        metadata.hidden_size,
        dtype,
        layer,
    )?;
    push_layer_block_rank3(
        blocks,
        WeightBlockRole::ExpertDownProjection,
        num_experts,
        metadata.hidden_size,
        moe_intermediate,
        dtype,
        layer,
    )?;
    if let Some(shared_intermediate) = metadata.shared_expert_intermediate_size {
        push_layer_block(
            blocks,
            WeightBlockRole::SharedExpertGateProjection,
            shared_intermediate,
            metadata.hidden_size,
            dtype,
            layer,
        )?;
        push_layer_block(
            blocks,
            WeightBlockRole::SharedExpertUpProjection,
            shared_intermediate,
            metadata.hidden_size,
            dtype,
            layer,
        )?;
        push_layer_block(
            blocks,
            WeightBlockRole::SharedExpertDownProjection,
            metadata.hidden_size,
            shared_intermediate,
            dtype,
            layer,
        )?;
        push_layer_block(
            blocks,
            WeightBlockRole::SharedExpertRouterProjection,
            1,
            metadata.hidden_size,
            dtype,
            layer,
        )?;
    }
    Ok(())
}

fn push_attention_bias_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    query_rows: usize,
    kv_hidden: usize,
    dtype: DType,
    layer: u32,
) -> Result<()> {
    if metadata.attention_qkv_bias {
        for (role, rows) in [
            (WeightBlockRole::QueryBias, query_rows),
            (WeightBlockRole::KeyBias, kv_hidden),
            (WeightBlockRole::ValueBias, kv_hidden),
        ] {
            push_layer_block(blocks, role, rows, 1, dtype, layer)?;
        }
    }
    if metadata.attention_output_bias {
        push_layer_block(
            blocks,
            WeightBlockRole::OutputBias,
            metadata.hidden_size,
            1,
            dtype,
            layer,
        )?;
    }
    Ok(())
}

fn push_layer_block(
    blocks: &mut Vec<WeightBlockSpec>,
    role: WeightBlockRole,
    rows: usize,
    cols: usize,
    dtype: DType,
    layer: u32,
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

fn push_layer_block_rank3(
    blocks: &mut Vec<WeightBlockSpec>,
    role: WeightBlockRole,
    depth: usize,
    rows: usize,
    cols: usize,
    dtype: DType,
    layer: u32,
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

pub(super) fn sum_weight_bytes(blocks: &[WeightBlockSpec]) -> Result<usize> {
    blocks.iter().try_fold(0usize, |acc, block| {
        acc.checked_add(block.bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: block.bytes,
                reason: "total weight byte count overflow".to_string(),
            })
    })
}
