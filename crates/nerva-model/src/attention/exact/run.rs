use std::time::Instant;

use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::attention::block::KvAttentionBlock;
use crate::attention::exact::ledger::record_attention_block_event;
use crate::attention::scratch::BlockwiseAttentionScratch;
use crate::common::math::dot;
use crate::common::shape::TransformerBlockShape;
use crate::common::validate::require_len;

pub fn exact_blockwise_attention_into(
    shape: TransformerBlockShape,
    query: &[f32],
    blocks: &[KvAttentionBlock<'_>],
    scratch: &mut BlockwiseAttentionScratch,
    output: &mut [f32],
    ledger: &mut TokenLedger,
) -> Result<()> {
    shape.validate()?;
    scratch.require_shape(shape)?;
    require_len("attention query", query.len(), shape.attention_hidden())?;
    require_len("attention output", output.len(), shape.attention_hidden())?;

    scratch.local_output.fill(0.0);
    scratch.global_m.fill(f32::NEG_INFINITY);
    scratch.global_l.fill(0.0);
    output.fill(0.0);

    let head_dim = shape.head_dim();
    let scale = (head_dim as f32).sqrt().recip();
    let mut total_tokens = 0usize;

    for block in blocks {
        let values = block
            .token_count
            .checked_mul(shape.kv_hidden())
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "KV attention block token count overflow".to_string(),
            })?;
        require_len("KV block keys", block.keys.len(), values)?;
        require_len("KV block values", block.values.len(), values)?;
        if block.token_count == 0 {
            continue;
        }
        total_tokens = total_tokens.checked_add(block.token_count).ok_or_else(|| {
            NervaError::InvalidArgument {
                reason: "KV attention total token count overflow".to_string(),
            }
        })?;
        let start = Instant::now();
        for head in 0..shape.heads {
            let head_start = head * head_dim;
            let head_end = head_start + head_dim;
            let kv_head = shape.kv_head_for_attention_head(head);
            scratch.local_output[head_start..head_end].fill(0.0);
            let mut local_m = f32::NEG_INFINITY;
            let mut local_l = 0.0f32;

            for token_index in 0..block.token_count {
                let token_start = token_index * shape.kv_hidden() + kv_head * head_dim;
                let token_end = token_start + head_dim;
                let score = dot(
                    &query[head_start..head_end],
                    &block.keys[token_start..token_end],
                ) * scale;
                let next_m = local_m.max(score);
                let old_scale = if local_l == 0.0 {
                    0.0
                } else {
                    (local_m - next_m).exp()
                };
                let new_scale = (score - next_m).exp();
                for (local, value) in scratch.local_output[head_start..head_end]
                    .iter_mut()
                    .zip(block.values[token_start..token_end].iter().copied())
                {
                    *local = *local * old_scale + value * new_scale;
                }
                local_l = local_l * old_scale + new_scale;
                local_m = next_m;
            }

            let global_m = scratch.global_m[head];
            let global_l = scratch.global_l[head];
            let next_m = global_m.max(local_m);
            let global_scale = if global_l == 0.0 {
                0.0
            } else {
                (global_m - next_m).exp()
            };
            let local_scale = (local_m - next_m).exp();
            for (global, local) in output[head_start..head_end]
                .iter_mut()
                .zip(scratch.local_output[head_start..head_end].iter().copied())
            {
                *global = *global * global_scale + local * local_scale;
            }
            scratch.global_l[head] = global_l * global_scale + local_l * local_scale;
            scratch.global_m[head] = next_m;
        }
        record_attention_block_event(shape, block, elapsed_ns(start), ledger);
    }

    if total_tokens == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "blockwise attention requires at least one KV token".to_string(),
        });
    }

    for head in 0..shape.heads {
        let normalizer = scratch.global_l[head];
        if normalizer == 0.0 || !normalizer.is_finite() {
            return Err(NervaError::InvalidArgument {
                reason: "blockwise attention produced invalid normalizer".to_string(),
            });
        }
        let head_start = head * head_dim;
        let head_end = head_start + head_dim;
        for value in &mut output[head_start..head_end] {
            *value /= normalizer;
        }
    }

    Ok(())
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().max(1).min(u64::MAX as u128) as u64
}
