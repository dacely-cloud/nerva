use nerva_core::types::dtype::DType;
use nerva_core::types::error::Result;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::shape::TransformerBlockShape;
use crate::precision::bits::{decode_f32_for_dtype, encode_f32_for_dtype, hash_u16s};
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::precision::scratch::PrecisionTransformerBlockScratch;
use crate::precision::smoke::summary::PrecisionDTypeBlockSmokeSummary;
use crate::precision::smoke::weights::smoke_weights;
use crate::reference::block::types::ReferenceTransformerBlock;
use crate::reference::scratch::types::TransformerBlockScratch;

pub(crate) fn run_dtype_smoke(
    dtype: DType,
    shape: TransformerBlockShape,
) -> Result<PrecisionDTypeBlockSmokeSummary> {
    let weights = smoke_weights();
    let block = PrecisionTransformerBlock::new_from_f32(
        dtype,
        shape,
        &weights.rms_attn_weight,
        &weights.rms_mlp_weight,
        &weights.w_q,
        &weights.w_k,
        &weights.w_v,
        &weights.w_o,
        &weights.w_gate,
        &weights.w_up,
        &weights.w_down,
        weights.rms_eps,
    )?;
    let reference = ReferenceTransformerBlock::new(
        shape,
        weights.rms_attn_weight.clone(),
        weights.rms_mlp_weight.clone(),
        weights.w_q.clone(),
        weights.w_k.clone(),
        weights.w_v.clone(),
        weights.w_o.clone(),
        weights.w_gate.clone(),
        weights.w_up.clone(),
        weights.w_down.clone(),
        weights.rms_eps,
    )?;

    let input_f32 = [1.0, 2.0];
    let input = [
        encode_f32_for_dtype(input_f32[0], dtype)?,
        encode_f32_for_dtype(input_f32[1], dtype)?,
    ];
    let mut scratch = PrecisionTransformerBlockScratch::new(shape)?;
    let mut output_bits = [0u16; 2];
    let mut ledger = TokenLedger::new(0);
    block.forward_into(&input, &mut scratch, &mut output_bits, &mut ledger)?;
    ledger.require_zero_hot_path_allocations()?;

    let mut reference_scratch = TransformerBlockScratch::new(shape)?;
    let mut reference_output = [0.0f32; 2];
    let mut reference_ledger = TokenLedger::new(0);
    reference.forward_into(
        &input_f32,
        &mut reference_scratch,
        &mut reference_output,
        &mut reference_ledger,
    )?;
    let expected_bits = [
        encode_f32_for_dtype(reference_output[0], dtype)?,
        encode_f32_for_dtype(reference_output[1], dtype)?,
    ];
    let output_f32 = [
        decode_f32_for_dtype(output_bits[0], dtype)?,
        decode_f32_for_dtype(output_bits[1], dtype)?,
    ];
    let max_abs_error = output_f32
        .iter()
        .zip(reference_output.iter())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);

    Ok(PrecisionDTypeBlockSmokeSummary {
        dtype,
        bit_parity: output_bits == expected_bits,
        output_bits,
        expected_bits,
        output_hash: hash_u16s(&output_bits),
        expected_hash: hash_u16s(&expected_bits),
        max_abs_error,
        hot_path_allocations: ledger.hot_path_allocations,
    })
}
