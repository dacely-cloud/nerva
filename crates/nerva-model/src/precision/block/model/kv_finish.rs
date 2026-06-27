use nerva_core::types::error::Result;

use crate::common::math::silu;
use crate::precision::block::model::PrecisionTransformerBlock;
use crate::precision::block::ops::{
    encode_vec_into, mat_vec_encoded_row_major, rms_norm_encoded_into,
};
use crate::precision::scratch::PrecisionTransformerBlockKvScratch;

pub(super) fn finish_attention_and_mlp(
    block: &PrecisionTransformerBlock,
    scratch: &mut PrecisionTransformerBlockKvScratch,
    output: &mut [u16],
) -> Result<()> {
    mat_vec_encoded_row_major(
        block.dtype,
        &block.w_o,
        &scratch.token.attn,
        &mut scratch.token.residual,
    )?;
    for (out, residual) in scratch
        .token
        .residual
        .iter_mut()
        .zip(scratch.token.input.iter().copied())
    {
        *out += residual;
    }
    rms_norm_encoded_into(
        block.dtype,
        &scratch.token.residual,
        &block.rms_mlp_weight,
        block.rms_eps,
        &mut scratch.token.mlp_norm,
    )?;
    mat_vec_encoded_row_major(
        block.dtype,
        &block.w_gate,
        &scratch.token.mlp_norm,
        &mut scratch.token.gate,
    )?;
    mat_vec_encoded_row_major(
        block.dtype,
        &block.w_up,
        &scratch.token.mlp_norm,
        &mut scratch.token.up,
    )?;
    for ((ff, gate), up) in scratch
        .token
        .ff
        .iter_mut()
        .zip(scratch.token.gate.iter().copied())
        .zip(scratch.token.up.iter().copied())
    {
        *ff = silu(gate) * up;
    }
    mat_vec_encoded_row_major(
        block.dtype,
        &block.w_down,
        &scratch.token.ff,
        &mut scratch.token.down,
    )?;
    for (out, mlp) in scratch
        .token
        .residual
        .iter_mut()
        .zip(scratch.token.down.iter().copied())
    {
        *out += mlp;
    }
    encode_vec_into(block.dtype, &scratch.token.residual, output)
}
