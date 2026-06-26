use crate::backend::ffi::{NervaCudaBackendContractResult, run_backend_contract_smoke};
use crate::backend::summary::CudaBackendContractSummary;
use crate::smoke::ffi::{CUDA_ERROR_NO_DEVICE, c_char_array_to_string};
use crate::smoke::status::SmokeStatus;

pub fn backend_contract_smoke(
    device_bytes: usize,
    pinned_bytes: usize,
) -> CudaBackendContractSummary {
    let mut out = NervaCudaBackendContractResult::default();
    let return_code =
        run_backend_contract_smoke(&mut out, device_bytes as u64, pinned_bytes as u64);
    let runtime_version = (out.runtime_version > 0).then_some(out.runtime_version);
    let observed_word = u32::try_from(out.observed_word)
        .ok()
        .filter(|value| *value != 0);

    if return_code == 0 && out.status == 0 && observed_word == Some(0x5a5a_5a5a) {
        return CudaBackendContractSummary {
            status: SmokeStatus::Ok,
            gpu_name: c_char_array_to_string(&out.gpu_name),
            driver_version: (out.driver_version > 0).then_some(out.driver_version),
            runtime_version,
            compute_capability_major: Some(out.compute_capability_major),
            compute_capability_minor: Some(out.compute_capability_minor),
            device_total_memory_bytes: usize::try_from(out.total_global_mem).ok(),
            pci_bus_id: c_char_array_to_string(&out.pci_bus_id),
            device_count: out.device_count,
            device_ordinal: out.device_ordinal,
            requested_device_bytes: usize::try_from(out.requested_device_bytes).unwrap_or(0),
            requested_pinned_bytes: usize::try_from(out.requested_pinned_bytes).unwrap_or(0),
            allocated_device_bytes: usize::try_from(out.allocated_device_bytes).unwrap_or(0),
            allocated_pinned_bytes: usize::try_from(out.allocated_pinned_bytes).unwrap_or(0),
            stream_creations: out.stream_creations,
            stream_destroys: out.stream_destroys,
            event_creations: out.event_creations,
            event_destroys: out.event_destroys,
            device_allocations: out.device_allocations,
            device_frees: out.device_frees,
            pinned_allocations: out.pinned_allocations,
            pinned_frees: out.pinned_frees,
            memset_bytes: out.memset_bytes,
            d2h_bytes: out.d2h_bytes,
            sync_calls: out.sync_calls,
            observed_word,
            hot_path_allocations: out.hot_path_allocations,
            error: None,
        };
    }

    let reason = format!(
        "CUDA backend contract failed: return_code={} status={} cuda_error={} device_count={} device_ordinal={} observed_word=0x{:08x} streams={}/{} events={}/{} device_allocs={}/{} pinned_allocs={}/{}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.device_ordinal,
        out.observed_word,
        out.stream_creations,
        out.stream_destroys,
        out.event_creations,
        out.event_destroys,
        out.device_allocations,
        out.device_frees,
        out.pinned_allocations,
        out.pinned_frees,
    );

    CudaBackendContractSummary {
        status: if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
            SmokeStatus::Unavailable
        } else {
            SmokeStatus::Failed
        },
        gpu_name: c_char_array_to_string(&out.gpu_name),
        driver_version: (out.driver_version > 0).then_some(out.driver_version),
        runtime_version,
        compute_capability_major: (out.compute_capability_major > 0)
            .then_some(out.compute_capability_major),
        compute_capability_minor: (out.compute_capability_major > 0)
            .then_some(out.compute_capability_minor),
        device_total_memory_bytes: usize::try_from(out.total_global_mem)
            .ok()
            .filter(|value| *value > 0),
        pci_bus_id: c_char_array_to_string(&out.pci_bus_id),
        device_count: out.device_count,
        device_ordinal: out.device_ordinal,
        requested_device_bytes: usize::try_from(out.requested_device_bytes).unwrap_or(0),
        requested_pinned_bytes: usize::try_from(out.requested_pinned_bytes).unwrap_or(0),
        allocated_device_bytes: usize::try_from(out.allocated_device_bytes).unwrap_or(0),
        allocated_pinned_bytes: usize::try_from(out.allocated_pinned_bytes).unwrap_or(0),
        stream_creations: out.stream_creations,
        stream_destroys: out.stream_destroys,
        event_creations: out.event_creations,
        event_destroys: out.event_destroys,
        device_allocations: out.device_allocations,
        device_frees: out.device_frees,
        pinned_allocations: out.pinned_allocations,
        pinned_frees: out.pinned_frees,
        memset_bytes: out.memset_bytes,
        d2h_bytes: out.d2h_bytes,
        sync_calls: out.sync_calls,
        observed_word,
        hot_path_allocations: out.hot_path_allocations,
        error: Some(reason),
    }
}
