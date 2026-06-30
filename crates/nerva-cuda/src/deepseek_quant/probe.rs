use crate::deepseek_quant::ffi::{NervaCudaDeepSeekQuantSmokeResult, run_deepseek_quant_smoke};
use crate::deepseek_quant::summary::CudaDeepSeekQuantSummary;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

pub fn deepseek_quant_smoke() -> CudaDeepSeekQuantSummary {
    let mut out = NervaCudaDeepSeekQuantSmokeResult::default();
    let return_code = run_deepseek_quant_smoke(&mut out);

    if return_code == 0
        && out.status == 0
        && out.fp8_rows == 3
        && out.fp8_cols == 4
        && out.fp8_block_rows == 2
        && out.fp8_block_cols == 2
        && out.mxfp4_rows == 2
        && out.mxfp4_packed_cols == 4
        && out.mxfp4_scale_packed_cols == 2
        && out.fp8_output_hash != 0
        && out.mxfp4_output_hash != 0
        && out.fp8_mismatches == 0
        && out.mxfp4_mismatches == 0
        && out.fp8_max_abs_diff == 0.0
        && out.mxfp4_max_abs_diff == 0.0
        && out.h2d_bytes > 0
        && out.d2h_bytes > 0
        && out.kernel_launches == 2
        && out.sync_calls == 1
        && out.hot_path_allocations == 0
    {
        return CudaDeepSeekQuantSummary {
            status: SmokeStatus::Ok,
            fp8_rows: out.fp8_rows,
            fp8_cols: out.fp8_cols,
            fp8_block_rows: out.fp8_block_rows,
            fp8_block_cols: out.fp8_block_cols,
            mxfp4_rows: out.mxfp4_rows,
            mxfp4_packed_cols: out.mxfp4_packed_cols,
            mxfp4_scale_packed_cols: out.mxfp4_scale_packed_cols,
            fp8_output_hash: out.fp8_output_hash,
            mxfp4_output_hash: out.mxfp4_output_hash,
            fp8_mismatches: out.fp8_mismatches,
            mxfp4_mismatches: out.mxfp4_mismatches,
            fp8_max_abs_diff: out.fp8_max_abs_diff,
            mxfp4_max_abs_diff: out.mxfp4_max_abs_diff,
            device_arena_bytes: out.device_arena_bytes,
            pinned_host_bytes: out.pinned_host_bytes,
            h2d_bytes: out.h2d_bytes,
            d2h_bytes: out.d2h_bytes,
            kernel_launches: out.kernel_launches,
            sync_calls: out.sync_calls,
            hot_path_allocations: out.hot_path_allocations,
            error: None,
        };
    }

    let reason = format!(
        "CUDA DeepSeek quant smoke failed: return_code={} status={} cuda_error={} device_count={} fp8_hash={} mxfp4_hash={} fp8_mismatches={} mxfp4_mismatches={} fp8_max_abs_diff={} mxfp4_max_abs_diff={} kernel_launches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.fp8_output_hash,
        out.mxfp4_output_hash,
        out.fp8_mismatches,
        out.mxfp4_mismatches,
        out.fp8_max_abs_diff,
        out.mxfp4_max_abs_diff,
        out.kernel_launches,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaDeepSeekQuantSummary::unavailable(reason)
    } else {
        CudaDeepSeekQuantSummary::failed(reason)
    }
}
