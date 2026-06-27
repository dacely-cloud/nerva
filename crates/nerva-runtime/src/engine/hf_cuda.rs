use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::block::forward::summary::CudaBlockForwardSummary;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_model::causal_lm::types::HfCausalLmModel;
use nerva_model::precision::scratch::PrecisionTransformerBlockScratch;

use crate::engine::cuda_block::run_precision_block_on_cuda;

#[derive(Clone, Debug)]
pub struct HfCudaLayerForwardSummary {
    pub layer_index: usize,
    pub token: TokenId,
    pub hidden: usize,
    pub output_hash: u64,
    pub expected_hash: u64,
    pub bit_parity: bool,
    pub hot_path_allocations: u64,
    pub cuda: CudaBlockForwardSummary,
}

impl HfCudaLayerForwardSummary {
    pub fn passed(&self) -> bool {
        self.cuda.status == SmokeStatus::Ok && self.bit_parity && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"layer_index\":{},\"token\":{},\"hidden\":{},\"output_hash\":{},\"expected_hash\":{},\"bit_parity\":{},\"hot_path_allocations\":{},\"cuda\":{}}}",
            self.layer_index,
            self.token.0,
            self.hidden,
            self.output_hash,
            self.expected_hash,
            self.bit_parity,
            self.hot_path_allocations,
            self.cuda.to_json(),
        )
    }
}

pub fn run_loaded_hf_layer_on_cuda(
    model: &HfCausalLmModel,
    layer_index: usize,
    token: TokenId,
) -> Result<HfCudaLayerForwardSummary> {
    let layer = model
        .layer(layer_index)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("HF CUDA layer index {layer_index} is out of range"),
        })?;
    let input = model.embedding_row(token)?;
    let shape = model.shape();
    let mut scratch = PrecisionTransformerBlockScratch::new(shape)?;
    let mut expected = vec![0u16; shape.hidden];
    let mut ledger = TokenLedger::new(0);
    layer.forward_into(input, &mut scratch, &mut expected, &mut ledger)?;
    ledger.require_zero_hot_path_allocations()?;

    let cuda = run_precision_block_on_cuda(layer, input, 0)?;
    let output_hash = hash_u16s(&cuda.output);
    let expected_hash = hash_u16s(&expected);
    Ok(HfCudaLayerForwardSummary {
        layer_index,
        token,
        hidden: shape.hidden,
        output_hash,
        expected_hash,
        bit_parity: cuda.output == expected,
        hot_path_allocations: ledger.hot_path_allocations + cuda.hot_path_allocations,
        cuda,
    })
}

fn hash_u16s(values: &[u16]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}
