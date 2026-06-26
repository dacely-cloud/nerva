use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

pub(crate) const CUDA_ERROR_NO_DEVICE: i32 = 100;
pub(crate) const SMOKE_WORD: u32 = 0x4e45_5256;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaDeviceSmokeResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) value: u32,
    pub(crate) device_count: i32,
    pub(crate) device_ordinal: i32,
    pub(crate) driver_version: i32,
    pub(crate) runtime_version: i32,
    pub(crate) compute_capability_major: i32,
    pub(crate) compute_capability_minor: i32,
    pub(crate) total_global_mem: u64,
    pub(crate) gpu_name: [c_char; 128],
    pub(crate) pci_bus_id: [c_char; 32],
}

impl Default for NervaCudaDeviceSmokeResult {
    fn default() -> Self {
        Self {
            status: -1,
            cuda_error: 0,
            value: 0,
            device_count: 0,
            device_ordinal: -1,
            driver_version: 0,
            runtime_version: 0,
            compute_capability_major: 0,
            compute_capability_minor: 0,
            total_global_mem: 0,
            gpu_name: [0; 128],
            pci_bus_id: [0; 32],
        }
    }
}

unsafe extern "C" {
    fn nerva_cuda_device_smoke(out: *mut NervaCudaDeviceSmokeResult) -> c_int;
}

pub(crate) fn run_device_smoke(out: &mut NervaCudaDeviceSmokeResult) -> c_int {
    unsafe { nerva_cuda_device_smoke(out) }
}

pub(crate) fn c_char_array_to_string<const N: usize>(value: &[c_char; N]) -> Option<String> {
    if value.first().copied().unwrap_or_default() == 0 {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(value.as_ptr()) }
            .to_string_lossy()
            .into_owned(),
    )
}
