use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;

use crate::hf::metadata::HfModelMetadata;
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
    push_layer_block(
        blocks,
        WeightBlockRole::QueryProjection,
        attention_hidden,
        hidden,
        dtype,
        layer,
    )?;
    push_qk_blocks(blocks, metadata, kv_hidden, dtype, layer)?;
    push_mlp_blocks(blocks, metadata, attention_hidden, kv_hidden, dtype, layer)?;
    push_attention_bias_blocks(blocks, metadata, attention_hidden, kv_hidden, dtype, layer)
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

fn push_mlp_blocks(
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
        (WeightBlockRole::MlpNorm, metadata.hidden_size, 1),
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

fn push_attention_bias_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    attention_hidden: usize,
    kv_hidden: usize,
    dtype: DType,
    layer: u32,
) -> Result<()> {
    if !metadata.attention_bias {
        return Ok(());
    }
    for (role, rows) in [
        (WeightBlockRole::QueryBias, attention_hidden),
        (WeightBlockRole::KeyBias, kv_hidden),
        (WeightBlockRole::ValueBias, kv_hidden),
        (WeightBlockRole::OutputBias, metadata.hidden_size),
    ] {
        push_layer_block(blocks, role, rows, 1, dtype, layer)?;
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

pub(super) fn sum_weight_bytes(blocks: &[WeightBlockSpec]) -> Result<usize> {
    blocks.iter().try_fold(0usize, |acc, block| {
        acc.checked_add(block.bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: block.bytes,
                reason: "total weight byte count overflow".to_string(),
            })
    })
}
