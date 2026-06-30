use crate::deepseek_quant::ffi::{
    NervaCudaDeepSeekQuantDequantResult, NervaCudaDeepSeekQuantFp8DequantRequest,
    NervaCudaDeepSeekQuantMxfp4DequantRequest, run_deepseek_quant_fp8_dequant,
    run_deepseek_quant_mxfp4_dequant,
};
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekDequantSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub rows: u32,
    pub cols: u32,
    pub block_rows: u32,
    pub block_cols: u32,
    pub output: Vec<f32>,
    pub output_hash: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

pub fn deepseek_fp8_e4m3fn_e8m0_dequant(
    weights: &[u8],
    scales: &[u8],
    rows: u32,
    cols: u32,
    block_rows: u32,
    block_cols: u32,
) -> CudaDeepSeekDequantSummary {
    let expected_values = rows as usize * cols as usize;
    if rows == 0 || cols == 0 || block_rows == 0 || block_cols == 0 {
        return failed_summary(
            rows,
            cols,
            block_rows,
            block_cols,
            vec![0.0; expected_values],
            "invalid DeepSeek FP8 dequant shape",
        );
    }
    let scale_cols = (cols as usize).div_ceil(block_cols as usize);
    let scale_rows = (rows as usize).div_ceil(block_rows as usize);
    if weights.len() != expected_values || scales.len() != scale_rows * scale_cols {
        return failed_summary(
            rows,
            cols,
            block_rows,
            block_cols,
            vec![0.0; expected_values],
            "invalid DeepSeek FP8 dequant shape",
        );
    }

    let mut output = vec![0.0f32; expected_values];
    let mut out = NervaCudaDeepSeekQuantDequantResult::default();
    let request = NervaCudaDeepSeekQuantFp8DequantRequest {
        rows,
        cols,
        block_rows,
        block_cols,
        weights: weights.as_ptr(),
        scales: scales.as_ptr(),
        output: output.as_mut_ptr(),
    };
    let return_code = run_deepseek_quant_fp8_dequant(&request, &mut out);
    summarize(return_code, out, output)
}

pub fn deepseek_mxfp4_e2m1_e8m0_dequant(
    packed: &[u8],
    scales: &[u8],
    rows: u32,
    packed_cols: u32,
    scale_packed_cols: u32,
) -> CudaDeepSeekDequantSummary {
    let packed_values = rows as usize * packed_cols as usize;
    if rows == 0 || packed_cols == 0 || scale_packed_cols == 0 {
        return failed_summary(
            rows,
            packed_cols * 2,
            1,
            scale_packed_cols * 2,
            vec![0.0; packed_values * 2],
            "invalid DeepSeek MXFP4 dequant shape",
        );
    }
    let scale_cols = (packed_cols as usize).div_ceil(scale_packed_cols as usize);
    if packed.len() != packed_values || scales.len() != rows as usize * scale_cols {
        return failed_summary(
            rows,
            packed_cols * 2,
            1,
            scale_packed_cols * 2,
            vec![0.0; packed_values * 2],
            "invalid DeepSeek MXFP4 dequant shape",
        );
    }

    let mut output = vec![0.0f32; packed_values * 2];
    let mut out = NervaCudaDeepSeekQuantDequantResult::default();
    let request = NervaCudaDeepSeekQuantMxfp4DequantRequest {
        rows,
        packed_cols,
        scale_packed_cols,
        packed: packed.as_ptr(),
        scales: scales.as_ptr(),
        output: output.as_mut_ptr(),
    };
    let return_code = run_deepseek_quant_mxfp4_dequant(&request, &mut out);
    summarize(return_code, out, output)
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekQuantDequantResult,
    output: Vec<f32>,
) -> CudaDeepSeekDequantSummary {
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
            "CUDA DeepSeek dequant failed: return_code={} status={} cuda_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.device_count
        ))
    };
    CudaDeepSeekDequantSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        rows: out.rows,
        cols: out.cols,
        block_rows: out.block_rows,
        block_cols: out.block_cols,
        output,
        output_hash: out.output_hash,
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
    rows: u32,
    cols: u32,
    block_rows: u32,
    block_cols: u32,
    output: Vec<f32>,
    error: impl Into<String>,
) -> CudaDeepSeekDequantSummary {
    CudaDeepSeekDequantSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        rows,
        cols,
        block_rows,
        block_cols,
        output,
        output_hash: 0,
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
