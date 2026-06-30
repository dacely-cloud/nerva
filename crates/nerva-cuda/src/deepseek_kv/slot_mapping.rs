use crate::deepseek_kv::ffi::{
    NervaCudaDeepSeekCompressedSlotMappingRequest, NervaCudaDeepSeekCompressedSlotMappingResult,
    run_deepseek_compressed_slot_mapping,
};
use crate::deepseek_kv::summary::CudaDeepSeekCompressedSlotMappingSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

pub fn deepseek_compressed_slot_mapping(
    query_start_loc: &[i32],
    seq_lens: &[i32],
    block_table: &[i32],
    block_table_stride: u32,
    block_size: u32,
    compress_ratio: u32,
) -> CudaDeepSeekCompressedSlotMappingSummary {
    let num_reqs = seq_lens.len();
    let num_tokens = query_start_loc
        .last()
        .copied()
        .filter(|value| *value >= 0)
        .unwrap_or(0) as usize;
    if query_start_loc.len() != num_reqs + 1
        || num_reqs == 0
        || num_tokens == 0
        || block_table_stride == 0
        || block_size == 0
        || compress_ratio == 0
        || num_reqs > u32::MAX as usize
        || num_tokens > u32::MAX as usize
        || block_table.len()
            < (num_reqs)
                .checked_mul(block_table_stride as usize)
                .unwrap_or(usize::MAX)
        || !query_start_loc.windows(2).all(|pair| pair[0] <= pair[1])
        || query_start_loc.iter().any(|value| *value < 0)
    {
        return failed_summary(
            num_tokens as u32,
            num_reqs as u32,
            block_table_stride,
            block_size,
            compress_ratio,
            Vec::new(),
            "invalid DeepSeek compressed slot mapping shape",
        );
    }

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
