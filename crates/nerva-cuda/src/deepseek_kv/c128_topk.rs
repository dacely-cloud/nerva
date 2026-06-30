use crate::deepseek_kv::ffi::{
    NervaCudaDeepSeekC128TopkMetadataRequest, NervaCudaDeepSeekC128TopkMetadataResult,
    run_deepseek_c128_topk_metadata,
};
use crate::deepseek_kv::summary::CudaDeepSeekC128TopkMetadataSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekC128TopkMetadataReference {
    pub global_decode: Vec<i32>,
    pub decode_lens: Vec<i32>,
    pub prefill_local: Vec<i32>,
    pub valid_decode_tokens: u32,
    pub decode_entries: u32,
    pub prefill_entries: u32,
}

pub fn deepseek_c128_topk_metadata_reference(
    positions: &[i64],
    num_decode_tokens: u32,
    token_to_req_indices: &[i32],
    block_table: &[i32],
    block_table_stride: u32,
    slot_mapping: &[i64],
    block_size: u32,
    compress_ratio: u32,
    max_compressed_tokens: u32,
) -> Result<DeepSeekC128TopkMetadataReference, String> {
    let shape = validate_c128_topk_metadata_shape(
        positions,
        num_decode_tokens,
        token_to_req_indices,
        block_table,
        block_table_stride,
        slot_mapping,
        block_size,
        compress_ratio,
        max_compressed_tokens,
    )?;
    let decode_values = shape.num_decode_tokens * shape.max_compressed_tokens;
    let prefill_values = shape.num_prefill_tokens * shape.max_compressed_tokens;
    let mut global_decode = vec![-1i32; decode_values];
    let mut decode_lens = vec![0i32; shape.num_decode_tokens];
    let mut prefill_local = vec![-1i32; prefill_values];

    for token_idx in 0..shape.num_tokens {
        let position = positions[token_idx];
        let num_compressed = if position >= 0 {
            ((position as u64 + 1) / u64::from(compress_ratio))
                .min(u64::from(max_compressed_tokens)) as usize
        } else {
            0
        };

        if token_idx < shape.num_decode_tokens {
            let valid_token = slot_mapping[token_idx] >= 0;
            let req_idx = token_to_req_indices[token_idx];
            let mut local_count = 0i32;
            for offset in 0..shape.max_compressed_tokens {
                let is_valid = offset < num_compressed;
                let mut slot = -1i32;
                if is_valid && req_idx >= 0 && (req_idx as usize) < shape.num_reqs {
                    let block_id = offset / shape.block_size;
                    let block_offset = offset % shape.block_size;
                    if block_id < shape.block_table_stride {
                        let block_number =
                            block_table[req_idx as usize * shape.block_table_stride + block_id];
                        if block_number >= 0 {
                            slot = block_number * block_size as i32 + block_offset as i32;
                        }
                    }
                }
                global_decode[token_idx * shape.max_compressed_tokens + offset] = slot;
                if is_valid {
                    local_count += 1;
                }
            }
            decode_lens[token_idx] = if valid_token { local_count } else { 0 };
        } else {
            let prefill_idx = token_idx - shape.num_decode_tokens;
            for offset in 0..shape.max_compressed_tokens {
                prefill_local[prefill_idx * shape.max_compressed_tokens + offset] =
                    if offset < num_compressed {
                        offset as i32
                    } else {
                        -1
                    };
            }
        }
    }

    let valid_decode_tokens = slot_mapping[..shape.num_decode_tokens]
        .iter()
        .filter(|slot| **slot >= 0)
        .count() as u32;
    let decode_entries = decode_lens.iter().map(|value| *value as u32).sum();
    let prefill_entries = prefill_local.iter().filter(|value| **value >= 0).count() as u32;

    Ok(DeepSeekC128TopkMetadataReference {
        global_decode,
        decode_lens,
        prefill_local,
        valid_decode_tokens,
        decode_entries,
        prefill_entries,
    })
}

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
    let Ok(shape) = validate_c128_topk_metadata_shape(
        positions,
        num_decode_tokens,
        token_to_req_indices,
        block_table,
        block_table_stride,
        slot_mapping,
        block_size,
        compress_ratio,
        max_compressed_tokens,
    ) else {
        return failed_summary(
            positions.len() as u32,
            num_decode_tokens,
            positions.len().saturating_sub(num_decode_tokens as usize) as u32,
            0,
            block_table_stride,
            block_size,
            compress_ratio,
            max_compressed_tokens,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            "invalid DeepSeek C128A top-k metadata shape",
        );
    };

    let mut global_decode = vec![-1i32; shape.num_decode_tokens * shape.max_compressed_tokens];
    let mut decode_lens = vec![0i32; shape.num_decode_tokens];
    let mut prefill_local = vec![-1i32; shape.num_prefill_tokens * shape.max_compressed_tokens];
    let request = NervaCudaDeepSeekC128TopkMetadataRequest {
        num_tokens: shape.num_tokens as u32,
        num_decode_tokens,
        num_reqs: shape.num_reqs as u32,
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

#[derive(Clone, Copy)]
struct C128TopkMetadataShape {
    num_tokens: usize,
    num_decode_tokens: usize,
    num_prefill_tokens: usize,
    num_reqs: usize,
    block_table_stride: usize,
    block_size: usize,
    max_compressed_tokens: usize,
}

#[allow(clippy::too_many_arguments)]
fn validate_c128_topk_metadata_shape(
    positions: &[i64],
    num_decode_tokens: u32,
    token_to_req_indices: &[i32],
    block_table: &[i32],
    block_table_stride: u32,
    slot_mapping: &[i64],
    block_size: u32,
    compress_ratio: u32,
    max_compressed_tokens: u32,
) -> Result<C128TopkMetadataShape, String> {
    let num_tokens = positions.len();
    let num_decode_tokens_usize = num_decode_tokens as usize;
    let num_reqs = token_to_req_indices
        .iter()
        .take(num_tokens)
        .copied()
        .filter(|value| *value >= 0)
        .max()
        .map_or(0usize, |value| value as usize + 1);
    let num_prefill_tokens = num_tokens.saturating_sub(num_decode_tokens_usize);
    let block_table_stride_usize = block_table_stride as usize;
    let block_table_len = num_reqs
        .checked_mul(block_table_stride_usize)
        .ok_or_else(|| "DeepSeek C128A block table shape overflow".to_string())?;

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
        || block_size > i32::MAX as u32
        || max_compressed_tokens > i32::MAX as u32
    {
        return Err("invalid DeepSeek C128A top-k metadata shape".to_string());
    }

    Ok(C128TopkMetadataShape {
        num_tokens,
        num_decode_tokens: num_decode_tokens_usize,
        num_prefill_tokens,
        num_reqs,
        block_table_stride: block_table_stride_usize,
        block_size: block_size as usize,
        max_compressed_tokens: max_compressed_tokens as usize,
    })
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
