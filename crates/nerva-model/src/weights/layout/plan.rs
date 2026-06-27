use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;

use crate::common::dtype::dtype_to_str;
use crate::hf::contract::validate_exact_runtime_contract;
use crate::hf::metadata::HfModelMetadata;
use crate::hf::validate::validate_hf_metadata;
use crate::weights::layout::entry::{WeightBlockRole, WeightBlockSpec};

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
        format!(
            "{{\"architecture\":\"{}\",\"dtype\":\"{}\",\"blocks\":{},\"layers\":{},\"total_weight_bytes\":{},\"per_layer_weight_bytes\":{},\"static_weight_bytes\":{},\"hidden_size\":{},\"head_dim\":{},\"kv_hidden_size\":{},\"tie_word_embeddings\":{}}}",
            self.metadata.architecture.as_str(),
            dtype_to_str(self.dtype),
            self.blocks.len(),
            self.metadata.num_hidden_layers,
            self.total_weight_bytes,
            self.per_layer_weight_bytes,
            self.static_weight_bytes,
            self.metadata.hidden_size,
            self.metadata.head_dim(),
            self.metadata.num_key_value_heads * self.metadata.head_dim(),
            self.metadata.tie_word_embeddings,
        )
    }
}

pub fn plan_hf_weight_layout(metadata: &HfModelMetadata) -> Result<HfWeightLayoutPlan> {
    validate_exact_runtime_contract(metadata)?;
    metadata.block_shape().validate()?;
    validate_hf_metadata(
        metadata.hidden_size,
        metadata.num_hidden_layers,
        metadata.num_attention_heads,
        metadata.num_key_value_heads,
        metadata.intermediate_size,
        metadata.vocab_size,
    )?;
    let dtype = metadata
        .torch_dtype
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "HF weight layout requires torch_dtype".to_string(),
        })?;
    let kv_hidden = metadata
        .num_key_value_heads
        .checked_mul(metadata.head_dim())
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: 0,
            reason: "KV hidden size overflow".to_string(),
        })?;

    let static_block_count = if metadata.tie_word_embeddings { 2 } else { 3 };
    let mut blocks =
        Vec::with_capacity(metadata.num_hidden_layers.saturating_mul(9) + static_block_count);
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
        push_layer_weight_blocks(&mut blocks, metadata, kv_hidden, dtype, layer)?;
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

pub(crate) fn push_layer_weight_blocks(
    blocks: &mut Vec<WeightBlockSpec>,
    metadata: &HfModelMetadata,
    kv_hidden: usize,
    dtype: DType,
    layer: u32,
) -> Result<()> {
    let hidden = metadata.hidden_size;
    let intermediate = metadata.intermediate_size;
    for (role, rows, cols) in [
        (WeightBlockRole::AttentionNorm, hidden, 1),
        (WeightBlockRole::QueryProjection, hidden, hidden),
        (WeightBlockRole::KeyProjection, kv_hidden, hidden),
        (WeightBlockRole::ValueProjection, kv_hidden, hidden),
        (WeightBlockRole::OutputProjection, hidden, hidden),
        (WeightBlockRole::MlpNorm, hidden, 1),
        (WeightBlockRole::GateProjection, intermediate, hidden),
        (WeightBlockRole::UpProjection, intermediate, hidden),
        (WeightBlockRole::DownProjection, hidden, intermediate),
    ] {
        blocks.push(WeightBlockSpec::new(
            role,
            Some(layer),
            rows,
            cols,
            dtype,
            MemoryTier::Dram,
        )?);
    }
    if metadata.attention_bias {
        for (role, rows) in [
            (WeightBlockRole::QueryBias, hidden),
            (WeightBlockRole::KeyBias, kv_hidden),
            (WeightBlockRole::ValueBias, kv_hidden),
            (WeightBlockRole::OutputBias, hidden),
        ] {
            blocks.push(WeightBlockSpec::new(
                role,
                Some(layer),
                rows,
                1,
                dtype,
                MemoryTier::Dram,
            )?);
        }
    }
    Ok(())
}

pub(crate) fn sum_weight_bytes(blocks: &[WeightBlockSpec]) -> Result<usize> {
    blocks.iter().try_fold(0usize, |acc, block| {
        acc.checked_add(block.bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: block.bytes,
                reason: "total weight byte count overflow".to_string(),
            })
    })
}
