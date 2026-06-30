use crate::deepseek_kv::ffi::{
    NervaCudaDeepSeekC128TopkMetadataRequest, NervaCudaDeepSeekC128TopkMetadataResult,
    run_deepseek_c128_topk_metadata,
};
use crate::deepseek_kv::summary::CudaDeepSeekC128TopkMetadataSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

pub fn deepseek_c128_topk_metadata(
    positions: &[i64],
    num_decode_tokens: u32,
    token_to_req_indices: &[i32],
    block_table: &[i32],
    block_table_stride: u32,
    slot_mapping: &[i64],
    block_size: u32,
    compress_ratio: u32,
    max_compressed_tokens: u32,
) -> CudaDeepSeekC128TopkMetadataSummary {
    let num_tokens = positions.len();
    let num_decode_tokens_usize = num_decode_tokens as usize;
    let num_reqs = token_to_req_indices
        .iter()
        .copied()
        .filter(|value| *value >= 0)
        .max()
        .map_or(0usize, |value| value as usize + 1);
    let num_prefill_tokens = num_tokens.saturating_sub(num_decode_tokens_usize);
    let block_table_len = num_reqs
        .checked_mul(block_table_stride as usize)
        .unwrap_or(usize::MAX);

    if num_tokens == 0
        || num_decode_tokens_usize > num_tokens
        || num_decode_tokens_usize == 0
        || num_prefill_tokens == 0
        || num_reqs == 0
        || num_tokens > u32::MAX as usize
        || num_reqs > u32::MAX as usize
        || token_to_req_indices.len() < num_tokens
        || slot_mapping.len() < num_tokens
        || block_table.len() < block_table_len
        || block_table_stride == 0
        || block_size == 0
        || compress_ratio == 0
        || max_compressed_tokens == 0
    {
        return failed_summary(
            num_tokens as u32,
            num_decode_tokens,
            num_prefill_tokens as u32,
            num_reqs as u32,
            block_table_stride,
            block_size,
            compress_ratio,
            max_compressed_tokens,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            "invalid DeepSeek C128A top-k metadata shape",
        );
    }

    let mut global_decode = vec![-1i32; num_decode_tokens_usize * max_compressed_tokens as usize];
    let mut decode_lens = vec![0i32; num_decode_tokens_usize];
    let mut prefill_local = vec![-1i32; num_prefill_tokens * max_compressed_tokens as usize];
    let request = NervaCudaDeepSeekC128TopkMetadataRequest {
        num_tokens: num_tokens as u32,
        num_decode_tokens,
        num_reqs: num_reqs as u32,
        block_table_stride,
        block_size,
        compress_ratio,
        max_compressed_tokens,
        positions: positions.as_ptr(),
        token_to_req_indices: token_to_req_indices.as_ptr(),
        block_table: block_table.as_ptr(),
        slot_mapping: slot_mapping.as_ptr(),
        global_decode: global_decode.as_mut_ptr(),
        decode_lens: decode_lens.as_mut_ptr(),
        prefill_local: prefill_local.as_mut_ptr(),
    };
    let mut out = NervaCudaDeepSeekC128TopkMetadataResult::default();
    let return_code = run_deepseek_c128_topk_metadata(&request, &mut out);
    summarize(return_code, out, global_decode, decode_lens, prefill_local)
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekC128TopkMetadataResult,
    global_decode: Vec<i32>,
    decode_lens: Vec<i32>,
    prefill_local: Vec<i32>,
) -> CudaDeepSeekC128TopkMetadataSummary {
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
            "CUDA DeepSeek C128A top-k metadata failed: return_code={} status={} cuda_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.device_count
        ))
    };
    CudaDeepSeekC128TopkMetadataSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        num_tokens: out.num_tokens,
        num_decode_tokens: out.num_decode_tokens,
        num_prefill_tokens: out.num_prefill_tokens,
        num_reqs: out.num_reqs,
        block_table_stride: out.block_table_stride,
        block_size: out.block_size,
        compress_ratio: out.compress_ratio,
        max_compressed_tokens: out.max_compressed_tokens,
        valid_decode_tokens: out.valid_decode_tokens,
        decode_entries: out.decode_entries,
        prefill_entries: out.prefill_entries,
        output_hash: out.output_hash,
        global_decode,
        decode_lens,
        prefill_local,
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

#[allow(clippy::too_many_arguments)]
fn failed_summary(
    num_tokens: u32,
    num_decode_tokens: u32,
    num_prefill_tokens: u32,
    num_reqs: u32,
    block_table_stride: u32,
    block_size: u32,
    compress_ratio: u32,
    max_compressed_tokens: u32,
    global_decode: Vec<i32>,
    decode_lens: Vec<i32>,
    prefill_local: Vec<i32>,
    error: impl Into<String>,
) -> CudaDeepSeekC128TopkMetadataSummary {
    CudaDeepSeekC128TopkMetadataSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        num_tokens,
        num_decode_tokens,
        num_prefill_tokens,
        num_reqs,
        block_table_stride,
        block_size,
        compress_ratio,
        max_compressed_tokens,
        valid_decode_tokens: 0,
        decode_entries: 0,
        prefill_entries: 0,
        output_hash: 0,
        global_decode,
        decode_lens,
        prefill_local,
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
