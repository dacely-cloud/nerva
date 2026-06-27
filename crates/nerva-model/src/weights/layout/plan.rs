use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;

use crate::common::dtype::dtype_to_str;
use crate::hf::contract::validate_exact_runtime_contract;
use crate::hf::metadata::HfModelMetadata;
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
        format!(
            "{{\"architecture\":\"{}\",\"dtype\":\"{}\",\"blocks\":{},\"layers\":{},\"total_weight_bytes\":{},\"per_layer_weight_bytes\":{},\"static_weight_bytes\":{},\"hidden_size\":{},\"attention_hidden_size\":{},\"head_dim\":{},\"kv_hidden_size\":{},\"tie_word_embeddings\":{}}}",
            self.metadata.architecture.as_str(),
            dtype_to_str(self.dtype),
            self.blocks.len(),
            self.metadata.num_hidden_layers,
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
    validate_exact_runtime_contract(metadata)?;
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
