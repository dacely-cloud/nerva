use crate::deepseek_kv::ffi::{
    run_deepseek_save_partial_states, NervaCudaDeepSeekSavePartialStatesRequest,
    NervaCudaDeepSeekSavePartialStatesResult,
};
use crate::deepseek_kv::summary::CudaDeepSeekSavePartialStatesSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekSavePartialStatesReference {
    pub state_cache: Vec<f32>,
    pub written_tokens: u32,
    pub skipped_tokens: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn deepseek_save_partial_states_reference(
    kv: &[f32],
    score: &[f32],
    ape: &[f32],
    positions: &[i64],
    slot_mapping: &[i64],
    block_size: u32,
    head_size: u32,
    state_width: u32,
    compress_ratio: u32,
    num_blocks: u32,
) -> Result<DeepSeekSavePartialStatesReference, String> {
    let shape = validate_save_partial_states_shape(
        kv,
        score,
        ape,
        positions,
        slot_mapping,
        block_size,
        head_size,
        state_width,
        compress_ratio,
        num_blocks,
    )?;
    let mut state_cache = vec![0.0f32; shape.state_values];
    let row_stride = shape.state_width * 2;
    let block_stride = shape.block_size * row_stride;
    let mut written_tokens = 0u32;
    let mut skipped_tokens = 0u32;

    for token_idx in 0..shape.num_tokens {
        let slot_id = slot_mapping[token_idx];
        if slot_id < 0 {
            skipped_tokens += 1;
            continue;
        }
        let slot_id = slot_id as usize;
        let block_idx = slot_id / shape.block_size;
        let pos_in_block = slot_id % shape.block_size;
        let base = block_idx * block_stride + pos_in_block * row_stride;
        let position = positions[token_idx];
        let ape_row = if position >= 0 {
            position as usize % shape.compress_ratio
        } else {
            0
        };

        for dim in 0..shape.head_size {
            state_cache[base + dim] = kv[token_idx * shape.head_size + dim];
            state_cache[base + shape.state_width + dim] =
                score[token_idx * shape.head_size + dim] + ape[ape_row * shape.head_size + dim];
        }
        written_tokens += 1;
    }

    Ok(DeepSeekSavePartialStatesReference {
        state_cache,
        written_tokens,
        skipped_tokens,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn deepseek_save_partial_states(
    kv: &[f32],
    score: &[f32],
    ape: &[f32],
    positions: &[i64],
    slot_mapping: &[i64],
    block_size: u32,
    head_size: u32,
    state_width: u32,
    compress_ratio: u32,
    num_blocks: u32,
) -> CudaDeepSeekSavePartialStatesSummary {
    let Ok(shape) = validate_save_partial_states_shape(
        kv,
        score,
        ape,
        positions,
        slot_mapping,
        block_size,
        head_size,
        state_width,
        compress_ratio,
        num_blocks,
    ) else {
        return failed_summary(
            positions.len() as u32,
            block_size,
            head_size,
            state_width,
            compress_ratio,
            num_blocks,
            Vec::new(),
            "invalid DeepSeek save partial states shape",
        );
    };

    let mut state_cache = vec![0.0f32; shape.state_values];
    let request = NervaCudaDeepSeekSavePartialStatesRequest {
        num_tokens: shape.num_tokens as u32,
        block_size,
        head_size,
        state_width,
        compress_ratio,
        num_blocks,
        kv: kv.as_ptr(),
        score: score.as_ptr(),
        ape: ape.as_ptr(),
        positions: positions.as_ptr(),
        slot_mapping: slot_mapping.as_ptr(),
        state_cache: state_cache.as_mut_ptr(),
    };
    let mut out = NervaCudaDeepSeekSavePartialStatesResult::default();
    let return_code = run_deepseek_save_partial_states(&request, &mut out);
    summarize(return_code, out, state_cache)
}

#[derive(Clone, Copy)]
struct SavePartialStatesShape {
    num_tokens: usize,
    block_size: usize,
    head_size: usize,
    state_width: usize,
    compress_ratio: usize,
    state_values: usize,
}

#[allow(clippy::too_many_arguments)]
fn validate_save_partial_states_shape(
    kv: &[f32],
    score: &[f32],
    ape: &[f32],
    positions: &[i64],
    slot_mapping: &[i64],
    block_size: u32,
    head_size: u32,
    state_width: u32,
    compress_ratio: u32,
    num_blocks: u32,
) -> Result<SavePartialStatesShape, String> {
    let num_tokens = positions.len();
    let token_values = num_tokens
        .checked_mul(head_size as usize)
        .ok_or_else(|| "DeepSeek save partial states token shape overflow".to_string())?;
    let ape_values = (compress_ratio as usize)
        .checked_mul(head_size as usize)
        .ok_or_else(|| "DeepSeek save partial states APE shape overflow".to_string())?;
    let state_values = (num_blocks as usize)
        .checked_mul(block_size as usize)
        .and_then(|value| value.checked_mul(state_width as usize))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| "DeepSeek save partial states cache shape overflow".to_string())?;
    let max_slot = (num_blocks as i64)
        .checked_mul(block_size as i64)
        .ok_or_else(|| "DeepSeek save partial states slot shape overflow".to_string())?;

    if num_tokens == 0
        || block_size == 0
        || head_size == 0
        || state_width < head_size
        || compress_ratio == 0
        || num_blocks == 0
        || num_tokens > u32::MAX as usize
        || kv.len() != token_values
        || score.len() != token_values
        || ape.len() != ape_values
        || slot_mapping.len() != num_tokens
        || slot_mapping
            .iter()
            .any(|slot| *slot < -1 || *slot >= max_slot)
    {
        return Err("invalid DeepSeek save partial states shape".to_string());
    }

    Ok(SavePartialStatesShape {
        num_tokens,
        block_size: block_size as usize,
        head_size: head_size as usize,
        state_width: state_width as usize,
        compress_ratio: compress_ratio as usize,
        state_values,
    })
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekSavePartialStatesResult,
    state_cache: Vec<f32>,
) -> CudaDeepSeekSavePartialStatesSummary {
    let status = if return_code == 0 && out.status == 0 {
        SmokeStatus::Ok
    } else if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        SmokeStatus::Unavailable
    } else {
        SmokeStatus::Failed
    };
    let error = if status == SmokeStatus::Ok {
        None
    } else {
        Some(format!(
            "CUDA DeepSeek save partial states failed: return_code={} status={} cuda_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.device_count
        ))
    };
    CudaDeepSeekSavePartialStatesSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        num_tokens: out.num_tokens,
        block_size: out.block_size,
        head_size: out.head_size,
        state_width: out.state_width,
        compress_ratio: out.compress_ratio,
        num_blocks: out.num_blocks,
        written_tokens: out.written_tokens,
        skipped_tokens: out.skipped_tokens,
        output_hash: out.output_hash,
        state_cache,
        device_arena_bytes: out.device_arena_bytes,
        pinned_host_bytes: out.pinned_host_bytes,
        h2d_bytes: out.h2d_bytes,
        d2h_bytes: out.d2h_bytes,
        kernel_launches: out.kernel_launches,
        sync_calls: out.sync_calls,
        hot_path_allocations: out.hot_path_allocations,
        error,
    }
}

fn failed_summary(
    num_tokens: u32,
    block_size: u32,
    head_size: u32,
    state_width: u32,
    compress_ratio: u32,
    num_blocks: u32,
    state_cache: Vec<f32>,
    error: impl Into<String>,
) -> CudaDeepSeekSavePartialStatesSummary {
    CudaDeepSeekSavePartialStatesSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        num_tokens,
        block_size,
        head_size,
        state_width,
        compress_ratio,
        num_blocks,
        written_tokens: 0,
        skipped_tokens: 0,
        output_hash: 0,
        state_cache,
        device_arena_bytes: 0,
        pinned_host_bytes: 0,
        h2d_bytes: 0,
        d2h_bytes: 0,
        kernel_launches: 0,
        sync_calls: 0,
        hot_path_allocations: 0,
        error: Some(error.into()),
    }
}
