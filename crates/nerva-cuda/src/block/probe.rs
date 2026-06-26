use crate::block::ffi::{
    CUDA_ERROR_NO_DEVICE, NervaCudaLoadedTinyBlockResult, NervaCudaTinyBlockResult,
    run_loaded_tiny_block_smoke, run_tiny_block_smoke,
};
use crate::block::summary::{CudaLoadedTinyBlockSummary, CudaTinyBlockSummary};
use crate::smoke::status::SmokeStatus;

pub fn tiny_block_smoke() -> CudaTinyBlockSummary {
    let mut out = NervaCudaTinyBlockResult::default();
    let return_code = run_tiny_block_smoke(&mut out);

    if return_code == 0
        && out.status == 0
        && out.hidden == 2
        && out.intermediate == 2
        && out.output_hash != 0
        && out.kernel_launches == 1
        && out.hot_path_allocations == 0
    {
        return CudaTinyBlockSummary {
            status: SmokeStatus::Ok,
            hidden: out.hidden,
            intermediate: out.intermediate,
            output: out.output,
            output_hash: out.output_hash,
            device_arena_bytes: out.device_arena_bytes,
            pinned_host_bytes: out.pinned_host_bytes,
            kernel_launches: out.kernel_launches,
            sync_calls: out.sync_calls,
            d2h_bytes: out.d2h_bytes,
            hot_path_allocations: out.hot_path_allocations,
            error: None,
        };
    }

    let reason = format!(
        "CUDA tiny block smoke failed: return_code={} status={} cuda_error={} device_count={} hidden={} intermediate={} output_hash={} kernel_launches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.hidden,
        out.intermediate,
        out.output_hash,
        out.kernel_launches,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaTinyBlockSummary::unavailable(reason)
    } else {
        CudaTinyBlockSummary::failed(reason)
    }
}

pub fn loaded_tiny_block_smoke() -> CudaLoadedTinyBlockSummary {
    let mut out = NervaCudaLoadedTinyBlockResult::default();
    let return_code = run_loaded_tiny_block_smoke(&mut out);

    if return_code == 0
        && out.status == 0
        && out.hidden == 2
        && out.intermediate == 2
        && out.output_hash != 0
        && out.resident_weight_bytes > 0
        && out.h2d_bytes >= out.resident_weight_bytes
        && out.kernel_launches == 1
        && out.hot_path_allocations == 0
    {
        return CudaLoadedTinyBlockSummary {
            status: SmokeStatus::Ok,
            hidden: out.hidden,
            intermediate: out.intermediate,
            output: out.output,
            output_hash: out.output_hash,
            resident_weight_bytes: out.resident_weight_bytes,
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
        "CUDA loaded tiny block smoke failed: return_code={} status={} cuda_error={} device_count={} hidden={} intermediate={} output_hash={} resident_weight_bytes={} h2d_bytes={} kernel_launches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.hidden,
        out.intermediate,
        out.output_hash,
        out.resident_weight_bytes,
        out.h2d_bytes,
        out.kernel_launches,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaLoadedTinyBlockSummary::unavailable(reason)
    } else {
        CudaLoadedTinyBlockSummary::failed(reason)
    }
}
