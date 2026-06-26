use std::os::raw::{c_char, c_int};

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaBackendContractResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) device_ordinal: i32,
    pub(crate) driver_version: i32,
    pub(crate) runtime_version: i32,
    pub(crate) compute_capability_major: i32,
    pub(crate) compute_capability_minor: i32,
    pub(crate) total_global_mem: u64,
    pub(crate) requested_device_bytes: u64,
    pub(crate) requested_pinned_bytes: u64,
    pub(crate) allocated_device_bytes: u64,
    pub(crate) allocated_pinned_bytes: u64,
    pub(crate) stream_creations: u64,
    pub(crate) stream_destroys: u64,
    pub(crate) event_creations: u64,
    pub(crate) event_destroys: u64,
    pub(crate) device_allocations: u64,
    pub(crate) device_frees: u64,
    pub(crate) pinned_allocations: u64,
    pub(crate) pinned_frees: u64,
    pub(crate) memset_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) sync_calls: u64,
    pub(crate) observed_word: u64,
    pub(crate) hot_path_allocations: u64,
    pub(crate) gpu_name: [c_char; 128],
    pub(crate) pci_bus_id: [c_char; 32],
}

impl Default for NervaCudaBackendContractResult {
    fn default() -> Self {
        Self {
            status: -1,
            cuda_error: 0,
            device_count: 0,
            device_ordinal: -1,
            driver_version: 0,
            runtime_version: 0,
            compute_capability_major: 0,
            compute_capability_minor: 0,
            total_global_mem: 0,
            requested_device_bytes: 0,
            requested_pinned_bytes: 0,
            allocated_device_bytes: 0,
            allocated_pinned_bytes: 0,
            stream_creations: 0,
            stream_destroys: 0,
            event_creations: 0,
            event_destroys: 0,
            device_allocations: 0,
            device_frees: 0,
            pinned_allocations: 0,
            pinned_frees: 0,
            memset_bytes: 0,
            d2h_bytes: 0,
            sync_calls: 0,
            observed_word: 0,
            hot_path_allocations: 0,
            gpu_name: [0; 128],
            pci_bus_id: [0; 32],
        }
    }
}

unsafe extern "C" {
    fn nerva_cuda_backend_contract_smoke(
        out: *mut NervaCudaBackendContractResult,
        device_bytes: u64,
        pinned_bytes: u64,
    ) -> c_int;
}

pub(crate) fn run_backend_contract_smoke(
    out: &mut NervaCudaBackendContractResult,
    device_bytes: u64,
    pinned_bytes: u64,
) -> c_int {
    unsafe { nerva_cuda_backend_contract_smoke(out, device_bytes, pinned_bytes) }
}
