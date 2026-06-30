use crate::deepseek_kv::ffi::{
    NervaCudaDeepSeekCompressedSlotMappingRequest, NervaCudaDeepSeekCompressedSlotMappingResult,
    run_deepseek_compressed_slot_mapping,
};
use crate::deepseek_kv::summary::CudaDeepSeekCompressedSlotMappingSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

pub fn deepseek_compressed_slot_mapping_reference(
    query_start_loc: &[i32],
    seq_lens: &[i32],
    block_table: &[i32],
    block_table_stride: u32,
    block_size: u32,
    compress_ratio: u32,
) -> Result<Vec<i64>, String> {
    let num_reqs = seq_lens.len();
    let num_tokens = validate_compressed_slot_mapping_shape(
        query_start_loc,
        seq_lens,
        block_table,
        block_table_stride,
        block_size,
        compress_ratio,
    )?;
    let block_table_stride = block_table_stride as usize;
    let block_size_i64 = i64::from(block_size);
    let compress_ratio = compress_ratio as i32;
    let block_size = block_size as i32;
    let mut output_slots = vec![-1i64; num_tokens];

    for req_idx in 0..num_reqs {
        let query_start = query_start_loc[req_idx];
        let query_end = query_start_loc[req_idx + 1];
        let query_len = query_end - query_start;
        let start_pos = seq_lens[req_idx] - query_len;
        for offset in 0..query_len {
            let output_idx = (query_start + offset) as usize;
            let pos = start_pos + offset;
            if pos >= 0 && (pos + 1) % compress_ratio == 0 {
                let compressed_pos = pos / compress_ratio;
                let block_id = compressed_pos / block_size;
                let block_offset = compressed_pos % block_size;
                if block_id >= 0 && (block_id as usize) < block_table_stride {
                    let block_number =
                        block_table[req_idx * block_table_stride + block_id as usize];
                    if block_number >= 0 {
                        output_slots[output_idx] =
                            i64::from(block_number) * block_size_i64 + i64::from(block_offset);
                    }
                }
            }
        }
    }

    Ok(output_slots)
}

pub fn deepseek_compressed_slot_mapping(
    query_start_loc: &[i32],
    seq_lens: &[i32],
    block_table: &[i32],
    block_table_stride: u32,
    block_size: u32,
    compress_ratio: u32,
) -> CudaDeepSeekCompressedSlotMappingSummary {
    let num_reqs = seq_lens.len();
    let Ok(num_tokens) = validate_compressed_slot_mapping_shape(
        query_start_loc,
        seq_lens,
        block_table,
        block_table_stride,
        block_size,
        compress_ratio,
    ) else {
        return failed_summary(
            query_start_loc
                .last()
                .copied()
                .filter(|value| *value >= 0)
                .unwrap_or(0) as u32,
            num_reqs as u32,
            block_table_stride,
            block_size,
            compress_ratio,
            Vec::new(),
            "invalid DeepSeek compressed slot mapping shape",
        );
    };

    let mut output_slots = vec![-1i64; num_tokens];
    let request = NervaCudaDeepSeekCompressedSlotMappingRequest {
        num_tokens: num_tokens as u32,
        num_reqs: num_reqs as u32,
        block_table_stride,
        block_size,
        compress_ratio,
        query_start_loc: query_start_loc.as_ptr(),
        seq_lens: seq_lens.as_ptr(),
        block_table: block_table.as_ptr(),
        output_slots: output_slots.as_mut_ptr(),
    };
    let mut out = NervaCudaDeepSeekCompressedSlotMappingResult::default();
    let return_code = run_deepseek_compressed_slot_mapping(&request, &mut out);
    summarize(return_code, out, output_slots)
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekCompressedSlotMappingResult,
    output_slots: Vec<i64>,
) -> CudaDeepSeekCompressedSlotMappingSummary {
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
            "CUDA DeepSeek compressed slot mapping failed: return_code={} status={} cuda_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.device_count
        ))
    };
    CudaDeepSeekCompressedSlotMappingSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        num_tokens: out.num_tokens,
        num_reqs: out.num_reqs,
        block_table_stride: out.block_table_stride,
        block_size: out.block_size,
        compress_ratio: out.compress_ratio,
        valid_slots: out.valid_slots,
        pad_slots: out.pad_slots,
        output_hash: out.output_hash,
        output_slots,
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

fn validate_compressed_slot_mapping_shape(
    query_start_loc: &[i32],
    seq_lens: &[i32],
    block_table: &[i32],
    block_table_stride: u32,
    block_size: u32,
    compress_ratio: u32,
) -> Result<usize, String> {
    let num_reqs = seq_lens.len();
    let num_tokens = query_start_loc
        .last()
        .copied()
        .filter(|value| *value >= 0)
        .unwrap_or(0) as usize;
    if query_start_loc.len() != num_reqs + 1 {
        return Err("query_start_loc length must equal num_reqs + 1".to_string());
    }
    if num_reqs == 0 || num_tokens == 0 {
        return Err("compressed slot mapping requires requests and tokens".to_string());
    }
    if block_table_stride == 0 || block_size == 0 || compress_ratio == 0 {
        return Err(
            "compressed slot mapping requires non-zero block stride, block size, and ratio"
                .to_string(),
        );
    }
    if block_table_stride > i32::MAX as u32
        || block_size > i32::MAX as u32
        || compress_ratio > i32::MAX as u32
    {
        return Err("compressed slot mapping parameters exceed i32 kernel limits".to_string());
    }
    if num_reqs > u32::MAX as usize || num_tokens > u32::MAX as usize {
        return Err("compressed slot mapping shape exceeds CUDA u32 limits".to_string());
    }
    let required_block_table = num_reqs
        .checked_mul(block_table_stride as usize)
        .ok_or_else(|| "compressed slot mapping block table shape overflow".to_string())?;
    if block_table.len() < required_block_table {
        return Err("compressed slot mapping block table is too small".to_string());
    }
    if !query_start_loc.windows(2).all(|pair| pair[0] <= pair[1])
        || query_start_loc.iter().any(|value| *value < 0)
    {
        return Err("compressed slot mapping query_start_loc must be monotonic".to_string());
    }
    Ok(num_tokens)
}

fn failed_summary(
    num_tokens: u32,
    num_reqs: u32,
    block_table_stride: u32,
    block_size: u32,
    compress_ratio: u32,
    output_slots: Vec<i64>,
    error: impl Into<String>,
) -> CudaDeepSeekCompressedSlotMappingSummary {
    CudaDeepSeekCompressedSlotMappingSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        num_tokens,
        num_reqs,
        block_table_stride,
        block_size,
        compress_ratio,
        valid_slots: 0,
        pad_slots: 0,
        output_hash: 0,
        output_slots,
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
