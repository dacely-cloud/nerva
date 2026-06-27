use nerva_core::types::dtype::DType;
use nerva_core::types::id::token::TokenId;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::shape::TransformerBlockShape;
use crate::hf::metadata::HfModelMetadata;
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::precision::scratch::PrecisionTransformerBlockScratch;
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
}

impl HfCausalLmContextMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LastTokenSeedOnly => "last_token_seed_only",
        }
    }
}

#[derive(Clone, Debug)]
pub struct HfCausalLmDecodeOutput {
    pub context_mode: HfCausalLmContextMode,
    pub prompt_tokens: Vec<TokenId>,
    pub seed_token: TokenId,
    pub generated_tokens: Vec<TokenId>,
    pub ledgers: Vec<TokenLedger>,
}

#[derive(Clone, Debug)]
pub struct HfCausalLmDecodeScratch {
    pub(crate) shape: TransformerBlockShape,
    pub(crate) vocab_size: usize,
    pub(crate) block: PrecisionTransformerBlockScratch,
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
        shape.validate()?;
        Ok(Self {
            shape,
            vocab_size,
            block: PrecisionTransformerBlockScratch::new(shape)?,
            hidden_bits: vec![0; shape.hidden],
            next_bits: vec![0; shape.hidden],
            decoded: vec![0.0; shape.hidden],
            normed: vec![0.0; shape.hidden],
            logits: vec![0.0; vocab_size],
        })
    }
}
