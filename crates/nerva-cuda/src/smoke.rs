//! CUDA runtime-backed smoke and capability probe.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

const SMOKE_WORD: u32 = 0x4e45_5256;
const CUDA_ERROR_NO_DEVICE: i32 = 100;

#[repr(C)]
#[derive(Copy, Clone)]
struct NervaCudaDeviceSmokeResult {
    status: i32,
    cuda_error: i32,
    value: u32,
    device_count: i32,
    device_ordinal: i32,
    driver_version: i32,
    runtime_version: i32,
    compute_capability_major: i32,
    compute_capability_minor: i32,
    total_global_mem: u64,
    gpu_name: [c_char; 128],
    pci_bus_id: [c_char; 32],
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SmokeStatus {
    Ok,
    Unavailable,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaSmokeSummary {
    pub status: SmokeStatus,
    pub gpu_name: Option<String>,
    pub driver_version: Option<i32>,
    pub runtime_version: Option<i32>,
    pub compute_capability_major: Option<i32>,
    pub compute_capability_minor: Option<i32>,
    pub device_total_memory_bytes: Option<usize>,
    pub pci_bus_id: Option<String>,
    pub device_arena_bytes: usize,
    pub pinned_host_bytes: usize,
    pub kernel_value: Option<u32>,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaSmokeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"gpu_name\":{},\"driver_version\":{},\"runtime_version\":{},\"compute_capability_major\":{},\"compute_capability_minor\":{},\"device_total_memory_bytes\":{},\"pci_bus_id\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"kernel_value\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            json_opt_str(self.gpu_name.as_deref()),
            json_opt_i32(self.driver_version),
            json_opt_i32(self.runtime_version),
            json_opt_i32(self.compute_capability_major),
            json_opt_i32(self.compute_capability_minor),
            json_opt_usize(self.device_total_memory_bytes),
            json_opt_str(self.pci_bus_id.as_deref()),
            self.device_arena_bytes,
            self.pinned_host_bytes,
            json_opt_u32(self.kernel_value),
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
    }

    pub(crate) fn unavailable(error: impl Into<String>, runtime_version: Option<i32>) -> Self {
        Self {
            status: SmokeStatus::Unavailable,
            gpu_name: None,
            driver_version: None,
            runtime_version,
            compute_capability_major: None,
            compute_capability_minor: None,
            device_total_memory_bytes: None,
            pci_bus_id: None,
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            kernel_value: None,
            hot_path_allocations: 0,
            error: Some(error.into()),
        }
    }

    pub(crate) fn failed(error: impl Into<String>, runtime_version: Option<i32>) -> Self {
        Self {
            status: SmokeStatus::Failed,
            gpu_name: None,
            driver_version: None,
            runtime_version,
            compute_capability_major: None,
            compute_capability_minor: None,
            device_total_memory_bytes: None,
            pci_bus_id: None,
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            kernel_value: None,
            hot_path_allocations: 0,
            error: Some(error.into()),
        }
    }
}

pub fn smoke() -> CudaSmokeSummary {
    let mut out = NervaCudaDeviceSmokeResult::default();
    let return_code = unsafe { nerva_cuda_device_smoke(&mut out) };
    let runtime_version = (out.runtime_version > 0).then_some(out.runtime_version);

    if return_code == 0 && out.status == 0 && out.value == SMOKE_WORD {
        return CudaSmokeSummary {
            status: SmokeStatus::Ok,
            gpu_name: c_char_array_to_string(&out.gpu_name),
            driver_version: (out.driver_version > 0).then_some(out.driver_version),
            runtime_version,
            compute_capability_major: Some(out.compute_capability_major),
            compute_capability_minor: Some(out.compute_capability_minor),
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

fn json_opt_i32(value: Option<i32>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_opt_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

fn json_opt_str(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("\"{}\"", escape_json(value)),
        None => "null".to_string(),
    }
}

pub(crate) fn escape_json(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
