use crate::deepseek_kv::ffi::{
    NervaCudaDeepSeekC4IndexerTopkRequest, NervaCudaDeepSeekC4IndexerTopkResult,
    run_deepseek_c4_indexer_topk,
};
use crate::deepseek_kv::summary::CudaDeepSeekC4IndexerTopkSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

pub fn deepseek_c4_indexer_topk_reference(
    query: &[f32],
    key_cache: &[f32],
    weights: &[f32],
    context_lens: &[i32],
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    topk_tokens: u32,
) -> Result<(Vec<i32>, Vec<f32>), String> {
    let shape = validate_c4_indexer_topk_shape(
        query,
        key_cache,
        weights,
        context_lens,
        num_tokens,
        num_heads,
        head_dim,
        topk_tokens,
    )?;
    let output_len = shape
        .num_tokens
        .checked_mul(shape.topk_tokens)
        .ok_or_else(|| "DeepSeek C4 indexer top-k output shape overflow".to_string())?;
    let mut topk_indices = vec![-1i32; output_len];
    let mut topk_scores = vec![f32::NEG_INFINITY; output_len];

    for token in 0..shape.num_tokens {
        let raw_context_len = context_lens[token];
        if raw_context_len <= 0 {
            continue;
        }
        let context_len = (raw_context_len as usize).min(shape.max_compressed_tokens);
        for slot in 0..context_len {
            let score = c4_indexer_score(token, slot, query, key_cache, weights, shape);
            insert_c4_topk(
                token,
                slot as i32,
                score,
                shape.topk_tokens,
                &mut topk_indices,
                &mut topk_scores,
            );
        }
    }

    Ok((topk_indices, topk_scores))
}

pub fn deepseek_c4_indexer_topk(
    query: &[f32],
    key_cache: &[f32],
    weights: &[f32],
    context_lens: &[i32],
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    topk_tokens: u32,
) -> CudaDeepSeekC4IndexerTopkSummary {
    let Ok(shape) = validate_c4_indexer_topk_shape(
        query,
        key_cache,
        weights,
        context_lens,
        num_tokens,
        num_heads,
        head_dim,
        topk_tokens,
    ) else {
        return failed_summary(
            num_tokens,
            num_heads,
            head_dim,
            0,
            topk_tokens,
            Vec::new(),
            Vec::new(),
            "invalid DeepSeek C4 indexer top-k shape",
        );
    };

    let output_len = shape.num_tokens * shape.topk_tokens;
    let mut topk_indices = vec![-1i32; output_len];
    let mut topk_scores = vec![f32::NEG_INFINITY; output_len];
    let request = NervaCudaDeepSeekC4IndexerTopkRequest {
        num_tokens,
        num_heads,
        head_dim,
        max_compressed_tokens: shape.max_compressed_tokens as u32,
        topk_tokens,
        query: query.as_ptr(),
        key_cache: key_cache.as_ptr(),
        weights: weights.as_ptr(),
        context_lens: context_lens.as_ptr(),
        topk_indices: topk_indices.as_mut_ptr(),
        topk_scores: topk_scores.as_mut_ptr(),
    };
    let mut out = NervaCudaDeepSeekC4IndexerTopkResult::default();
    let return_code = run_deepseek_c4_indexer_topk(&request, &mut out);
    summarize(return_code, out, topk_indices, topk_scores)
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekC4IndexerTopkResult,
    topk_indices: Vec<i32>,
    topk_scores: Vec<f32>,
) -> CudaDeepSeekC4IndexerTopkSummary {
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
            "CUDA DeepSeek C4 indexer top-k failed: return_code={} status={} cuda_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.device_count
        ))
    };
    CudaDeepSeekC4IndexerTopkSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        num_tokens: out.num_tokens,
        num_heads: out.num_heads,
        head_dim: out.head_dim,
        max_compressed_tokens: out.max_compressed_tokens,
        topk_tokens: out.topk_tokens,
        valid_tokens: out.valid_tokens,
        selected_entries: out.selected_entries,
        output_hash: out.output_hash,
        topk_indices,
        topk_scores,
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

#[derive(Clone, Copy)]
struct C4IndexerTopkShape {
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    max_compressed_tokens: usize,
    topk_tokens: usize,
}

fn validate_c4_indexer_topk_shape(
    query: &[f32],
    key_cache: &[f32],
    weights: &[f32],
    context_lens: &[i32],
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    topk_tokens: u32,
) -> Result<C4IndexerTopkShape, String> {
    let num_tokens_usize = num_tokens as usize;
    let num_heads_usize = num_heads as usize;
    let head_dim_usize = head_dim as usize;
    let topk_tokens_usize = topk_tokens as usize;
    let max_compressed_tokens = if head_dim_usize == 0 {
        0usize
    } else {
        key_cache.len() / head_dim_usize
    };
    let query_len = num_tokens_usize
        .checked_mul(num_heads_usize)
        .and_then(|len| len.checked_mul(head_dim_usize))
        .ok_or_else(|| "DeepSeek C4 indexer query shape overflow".to_string())?;
    let weights_len = num_tokens_usize
        .checked_mul(num_heads_usize)
        .ok_or_else(|| "DeepSeek C4 indexer weight shape overflow".to_string())?;
    let _output_len = num_tokens_usize
        .checked_mul(topk_tokens_usize)
        .ok_or_else(|| "DeepSeek C4 indexer output shape overflow".to_string())?;

    if num_tokens == 0
        || num_heads == 0
        || head_dim == 0
        || topk_tokens == 0
        || max_compressed_tokens == 0
        || max_compressed_tokens > u32::MAX as usize
        || context_lens.len() < num_tokens_usize
        || query.len() < query_len
        || weights.len() < weights_len
        || key_cache.len() != max_compressed_tokens * head_dim_usize
    {
        return Err("invalid DeepSeek C4 indexer top-k shape".to_string());
    }

    Ok(C4IndexerTopkShape {
        num_tokens: num_tokens_usize,
        num_heads: num_heads_usize,
        head_dim: head_dim_usize,
        max_compressed_tokens,
        topk_tokens: topk_tokens_usize,
    })
}

fn c4_indexer_score(
    token: usize,
    slot: usize,
    query: &[f32],
    key_cache: &[f32],
    weights: &[f32],
    shape: C4IndexerTopkShape,
) -> f32 {
    let mut score = 0.0f32;
    for head in 0..shape.num_heads {
        let head_weight = weights[token * shape.num_heads + head];
        let query_base = (token * shape.num_heads + head) * shape.head_dim;
        let key_base = slot * shape.head_dim;
        let mut dot = 0.0f32;
        for dim in 0..shape.head_dim {
            dot += query[query_base + dim] * key_cache[key_base + dim];
        }
        score += head_weight * dot;
    }
    score
}

fn insert_c4_topk(
    token: usize,
    slot: i32,
    score: f32,
    topk_tokens: usize,
    topk_indices: &mut [i32],
    topk_scores: &mut [f32],
) {
    let output_base = token * topk_tokens;
    for rank in 0..topk_tokens {
        let output_idx = output_base + rank;
        if !c4_indexer_score_is_better(
            score,
            slot,
            topk_scores[output_idx],
            topk_indices[output_idx],
        ) {
            continue;
        }
        for shift in (rank + 1..topk_tokens).rev() {
            topk_indices[output_base + shift] = topk_indices[output_base + shift - 1];
            topk_scores[output_base + shift] = topk_scores[output_base + shift - 1];
        }
        topk_indices[output_idx] = slot;
        topk_scores[output_idx] = score;
        break;
    }
}

fn c4_indexer_score_is_better(candidate: f32, slot: i32, current: f32, current_slot: i32) -> bool {
    if !candidate.is_finite() {
        return false;
    }
    if current_slot < 0 {
        return true;
    }
    candidate > current || (candidate == current && slot >= 0 && slot < current_slot)
}

#[allow(clippy::too_many_arguments)]
fn failed_summary(
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    max_compressed_tokens: u32,
    topk_tokens: u32,
    topk_indices: Vec<i32>,
    topk_scores: Vec<f32>,
    error: impl Into<String>,
) -> CudaDeepSeekC4IndexerTopkSummary {
    CudaDeepSeekC4IndexerTopkSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        num_tokens,
        num_heads,
        head_dim,
        max_compressed_tokens,
        topk_tokens,
        valid_tokens: 0,
        selected_entries: 0,
        output_hash: 0,
        topk_indices,
        topk_scores,
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
