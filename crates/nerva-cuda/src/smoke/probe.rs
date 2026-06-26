use crate::smoke::ffi::{
    CUDA_ERROR_NO_DEVICE, NervaCudaDeviceSmokeResult, SMOKE_WORD, c_char_array_to_string,
    run_device_smoke,
};
use crate::smoke::status::SmokeStatus;
use crate::smoke::summary::CudaSmokeSummary;

pub fn smoke() -> CudaSmokeSummary {
    let mut out = NervaCudaDeviceSmokeResult::default();
    let return_code = run_device_smoke(&mut out);
    let runtime_version = (out.runtime_version > 0).then_some(out.runtime_version);

    if return_code == 0 && out.status == 0 && out.value == SMOKE_WORD {
        return CudaSmokeSummary {
            status: SmokeStatus::Ok,
            gpu_name: c_char_array_to_string(&out.gpu_name),
            driver_version: (out.driver_version > 0).then_some(out.driver_version),
            runtime_version,
            compute_capability_major: Some(out.compute_capability_major),
            compute_capability_minor: Some(out.compute_capability_minor),
            posix_fd_handle_supported: attr_bool(out.posix_fd_handle_supported),
            vmm_posix_fd_export_verified: attr_bool(out.vmm_posix_fd_export_verified),
            gpu_direct_rdma_supported: attr_bool(out.gpu_direct_rdma_supported),
            gpu_direct_rdma_with_cuda_vmm_supported: attr_bool(
                out.gpu_direct_rdma_with_cuda_vmm_supported,
            ),
            device_total_memory_bytes: usize::try_from(out.total_global_mem).ok(),
            pci_bus_id: c_char_array_to_string(&out.pci_bus_id),
            device_arena_bytes: 4,
            pinned_host_bytes: 4,
            kernel_value: Some(out.value),
            hot_path_allocations: 0,
            error: None,
        };
    }

    let reason = format!(
        "CUDA runtime smoke failed: return_code={} status={} cuda_error={} device_count={} device_ordinal={} value=0x{:08x}",
        return_code, out.status, out.cuda_error, out.device_count, out.device_ordinal, out.value,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaSmokeSummary::unavailable(reason, runtime_version)
    } else {
        CudaSmokeSummary::failed(reason, runtime_version)
    }
}

fn attr_bool(value: i32) -> Option<bool> {
    match value {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}
