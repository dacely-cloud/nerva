use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::shape::TransformerBlockShape;
use crate::hf::metadata::HfModelMetadata;
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::precision::scratch::{
    PrecisionTransformerBlockKvScratch, PrecisionTransformerBlockScratch,
};
use crate::weights::manifest::HfTensorManifest;
use crate::weights::safetensors::shard::SafetensorsShardPlan;

#[derive(Clone, Debug)]
pub struct HfCausalLmModel {
    pub(crate) metadata: HfModelMetadata,
    pub(crate) dtype: DType,
    pub(crate) layers: Vec<PrecisionTransformerBlock>,
    pub(crate) embeddings: Vec<u16>,
    pub(crate) final_norm: Vec<u16>,
    pub(crate) lm_head: Vec<u16>,
    pub(crate) rms_eps: f32,
}

impl HfCausalLmModel {
    pub fn metadata(&self) -> &HfModelMetadata {
        &self.metadata
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> TransformerBlockShape {
        self.metadata.block_shape()
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, index: usize) -> Option<&PrecisionTransformerBlock> {
        self.layers.get(index)
    }

    pub fn embedding_row(&self, token: TokenId) -> Result<&[u16]> {
        let hidden = self.metadata.hidden_size;
        let start =
            (token.0 as usize)
                .checked_mul(hidden)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: "HF causal LM embedding row offset overflow".to_string(),
                })?;
        let end = start
            .checked_add(hidden)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "HF causal LM embedding row end overflow".to_string(),
            })?;
        self.embeddings
            .get(start..end)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "HF causal LM embedding token is outside vocabulary".to_string(),
            })
    }
}

#[derive(Clone, Debug)]
pub struct HfCausalLmLoadSummary {
    pub manifest: HfTensorManifest,
    pub shard_plan: SafetensorsShardPlan,
    pub tensors_loaded: usize,
    pub bytes_loaded: usize,
    pub data_hash: u64,
    pub tied_lm_head: bool,
}

#[derive(Clone, Debug)]
pub struct HfCausalLmLoaded {
    pub model: HfCausalLmModel,
    pub summary: HfCausalLmLoadSummary,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfCausalLmContextMode {
    LastTokenSeedOnly,
    PromptPrefillKvDecode,
}

impl HfCausalLmContextMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LastTokenSeedOnly => "last_token_seed_only",
            Self::PromptPrefillKvDecode => "prompt_prefill_kv_decode",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfCausalLmStopReason {
    MaxSteps,
    EosToken,
}

impl HfCausalLmStopReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MaxSteps => "max_steps",
            Self::EosToken => "eos_token",
        }
    }
}

#[derive(Clone, Debug)]
pub struct HfCausalLmDecodeOutput {
    pub context_mode: HfCausalLmContextMode,
    pub stop_reason: HfCausalLmStopReason,
    pub prompt_tokens: Vec<TokenId>,
    pub seed_token: TokenId,
    pub generated_tokens: Vec<TokenId>,
    pub ledgers: Vec<TokenLedger>,
}

#[derive(Clone, Debug)]
pub struct HfCausalLmDecodeScratch {
    pub(crate) shape: TransformerBlockShape,
    pub(crate) vocab_size: usize,
    pub(crate) max_context_tokens: usize,
    pub(crate) block: PrecisionTransformerBlockScratch,
    pub(crate) kv_layers: Vec<PrecisionTransformerBlockKvScratch>,
    pub(crate) sequence_bits: Vec<u16>,
    pub(crate) sequence_next_bits: Vec<u16>,
    pub(crate) hidden_bits: Vec<u16>,
    pub(crate) next_bits: Vec<u16>,
    pub(crate) decoded: Vec<f32>,
    pub(crate) normed: Vec<f32>,
    pub(crate) logits: Vec<f32>,
}

impl HfCausalLmDecodeScratch {
    pub fn new(
        shape: TransformerBlockShape,
        vocab_size: usize,
    ) -> nerva_core::types::error::Result<Self> {
        Self::with_context_capacity(shape, vocab_size, 0, 0)
    }

    pub fn new_with_context(
        shape: TransformerBlockShape,
        vocab_size: usize,
        layer_count: usize,
        max_context_tokens: usize,
    ) -> nerva_core::types::error::Result<Self> {
        Self::with_context_capacity(shape, vocab_size, layer_count, max_context_tokens)
    }

    fn with_context_capacity(
        shape: TransformerBlockShape,
        vocab_size: usize,
        layer_count: usize,
        max_context_tokens: usize,
    ) -> nerva_core::types::error::Result<Self> {
        shape.validate()?;
        let mut kv_layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            kv_layers.push(PrecisionTransformerBlockKvScratch::new(
                shape,
                max_context_tokens,
            )?);
        }
        let context_values = max_context_tokens * shape.hidden;
        Ok(Self {
            shape,
            vocab_size,
            max_context_tokens,
            block: PrecisionTransformerBlockScratch::new(shape)?,
            kv_layers,
            sequence_bits: vec![0; context_values],
            sequence_next_bits: vec![0; context_values],
            hidden_bits: vec![0; shape.hidden],
            next_bits: vec![0; shape.hidden],
            decoded: vec![0.0; shape.hidden],
            normed: vec![0.0; shape.hidden],
            logits: vec![0.0; vocab_size],
        })
    }
}
