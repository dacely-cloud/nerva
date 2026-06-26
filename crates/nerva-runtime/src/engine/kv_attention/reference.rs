use nerva_core::types::error::Result;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_model::attention::block::KvAttentionBlock;
use nerva_model::attention::exact::run::exact_blockwise_attention_into;
use nerva_model::attention::scratch::BlockwiseAttentionScratch;
use nerva_model::common::shape::TransformerBlockShape;

use crate::engine::kv_attention::compare::hash_f32s;

pub(super) fn reference_attention(
    shape: TransformerBlockShape,
    query: &[f32],
    dram_keys: &[f32],
    dram_values: &[f32],
    vram_keys: &[f32],
    vram_values: &[f32],
) -> Result<([f32; 2], u64)> {
    let mut reference_keys = Vec::with_capacity(dram_keys.len() + vram_keys.len());
    reference_keys.extend_from_slice(dram_keys);
    reference_keys.extend_from_slice(vram_keys);
    let mut reference_values = Vec::with_capacity(dram_values.len() + vram_values.len());
    reference_values.extend_from_slice(dram_values);
    reference_values.extend_from_slice(vram_values);

    let reference_block =
        KvAttentionBlock::new(&reference_keys, &reference_values, 4, MemoryTier::Dram);
    let mut scratch = BlockwiseAttentionScratch::new(shape)?;
    let mut reference = [0.0; 2];
    let mut ledger = TokenLedger::new(0);
    exact_blockwise_attention_into(
        shape,
        query,
        &[reference_block],
        &mut scratch,
        &mut reference,
        &mut ledger,
    )?;
    let hash = hash_f32s(&reference);
    Ok((reference, hash))
}
