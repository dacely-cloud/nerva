use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::shape::TransformerBlockShape;
use crate::hf::metadata::HfModelMetadata;
use crate::precision::block::gdn::PrecisionGatedDeltaNetMoeBlock;
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::precision::block::moe::PrecisionMoeTransformerBlock;
use crate::precision::scratch::{
    PrecisionTransformerBlockKvScratch, PrecisionTransformerBlockScratch,
};
use crate::weights::manifest::HfTensorManifest;
use crate::weights::safetensors::shard::SafetensorsShardPlan;

#[derive(Clone, Debug)]
pub struct HfCausalLmModel {
    pub(crate) metadata: HfModelMetadata,
    pub(crate) dtype: DType,
    pub(crate) layers: Vec<HfCausalLmLayer>,
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
        self.layers.get(index).and_then(HfCausalLmLayer::as_dense)
    }

    pub fn causal_layer(&self, index: usize) -> Option<&HfCausalLmLayer> {
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
pub enum HfCausalLmLayer {
    Dense(PrecisionTransformerBlock),
    SparseMoe(PrecisionMoeTransformerBlock),
    GatedDeltaNetMoe(PrecisionGatedDeltaNetMoeBlock),
}

impl HfCausalLmLayer {
    pub fn as_dense(&self) -> Option<&PrecisionTransformerBlock> {
        match self {
            Self::Dense(layer) => Some(layer),
            Self::SparseMoe(_) => None,
            Self::GatedDeltaNetMoe(_) => None,
        }
    }

    pub fn rope_theta(&self) -> Option<f32> {
        match self {
            Self::Dense(layer) => layer.rope_theta(),
            Self::SparseMoe(layer) => layer.rope_theta(),
            Self::GatedDeltaNetMoe(_) => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn forward_into(
        &self,
        input: &[u16],
        scratch: &mut PrecisionTransformerBlockScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        self.forward_with_token_into(input, None, scratch, output, ledger)
    }

    pub(crate) fn forward_with_token_into(
        &self,
        input: &[u16],
        route_token: Option<TokenId>,
        scratch: &mut PrecisionTransformerBlockScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        match self {
            Self::Dense(layer) => layer.forward_into(input, scratch, output, ledger),
            Self::SparseMoe(layer) => {
                layer.forward_with_token_into(input, route_token, scratch, output, ledger)
            }
            Self::GatedDeltaNetMoe(layer) => layer.forward_into(input, scratch, output, ledger),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn forward_prefill_sequence_into(
        &self,
        input: &[u16],
        token_count: usize,
        scratch: &mut PrecisionTransformerBlockKvScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        self.forward_prefill_sequence_with_tokens_into(
            input,
            token_count,
            None,
            scratch,
            output,
            ledger,
        )
    }

    pub(crate) fn forward_prefill_sequence_with_tokens_into(
        &self,
        input: &[u16],
        token_count: usize,
        route_tokens: Option<&[TokenId]>,
        scratch: &mut PrecisionTransformerBlockKvScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        match self {
            Self::Dense(layer) => {
                layer.forward_prefill_sequence_into(input, token_count, scratch, output, ledger)
            }
            Self::SparseMoe(layer) => layer.forward_prefill_sequence_with_tokens_into(
                input,
                token_count,
                route_tokens,
                scratch,
                output,
                ledger,
            ),
            Self::GatedDeltaNetMoe(layer) => {
                layer.forward_prefill_sequence_into(input, token_count, scratch, output, ledger)
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn forward_decode_with_kv_into(
        &self,
        input: &[u16],
        scratch: &mut PrecisionTransformerBlockKvScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        self.forward_decode_with_token_kv_into(input, None, scratch, output, ledger)
    }

    pub(crate) fn forward_decode_with_token_kv_into(
        &self,
        input: &[u16],
        route_token: Option<TokenId>,
        scratch: &mut PrecisionTransformerBlockKvScratch,
        output: &mut [u16],
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        match self {
            Self::Dense(layer) => layer.forward_decode_with_kv_into(input, scratch, output, ledger),
            Self::SparseMoe(layer) => {
                layer.forward_decode_with_token_kv_into(input, route_token, scratch, output, ledger)
            }
            Self::GatedDeltaNetMoe(layer) => {
                layer.forward_decode_with_kv_into(input, scratch, output, ledger)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct HfCausalLmLoadSummary {
    pub manifest: HfTensorManifest,
    pub shard_plan: SafetensorsShardPlan,
    pub tensors_loaded: usize,
    pub bytes_loaded: usize,
    pub data_hash: u64,
    pub data_hash_available: bool,
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
