use std::os::raw::c_char;

use crate::smoke::ffi::c_char_array_to_string;
use crate::smoke::probe::smoke;
use crate::smoke::status::SmokeStatus;
use crate::smoke::summary::CudaSmokeSummary;

#[test]
fn unavailable_summary_is_valid_shape() {
    let summary = CudaSmokeSummary::unavailable("no cuda", Some(13_010));
    let json = summary.to_json();
    assert!(json.contains("\"status\":\"unavailable\""));
    assert!(json.contains("\"runtime_version\":13010"));
    assert!(json.contains("\"compute_capability_major\":null"));
    assert!(json.contains("\"compute_capability_minor\":null"));
    assert!(json.contains("\"posix_fd_handle_supported\":null"));
    assert!(json.contains("\"vmm_posix_fd_export_verified\":null"));
    assert!(json.contains("\"gpu_direct_rdma_supported\":null"));
    assert!(json.contains("\"gpu_direct_rdma_with_cuda_vmm_supported\":null"));
    assert!(json.contains("\"device_total_memory_bytes\":null"));
    assert!(json.contains("\"device_free_memory_bytes\":null"));
    assert!(json.contains("\"pci_bus_id\":null"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn c_char_array_conversion_handles_empty_and_terminated_values() {
    let empty = [0 as c_char; 8];
    assert_eq!(c_char_array_to_string(&empty), None);

    let mut value = [0 as c_char; 8];
    value[0] = b'R' as c_char;
    value[1] = b'T' as c_char;
    value[2] = b'X' as c_char;
    assert_eq!(c_char_array_to_string(&value).as_deref(), Some("RTX"));
}

#[test]
fn cuda_smoke_is_repeatable_when_device_is_available() {
    let first = smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.kernel_value, Some(0x4e45_5256));
    assert_eq!(second.hot_path_allocations, 0);
    assert!(second.posix_fd_handle_supported.is_some());
    assert!(second.vmm_posix_fd_export_verified.is_some());
    assert!(second.gpu_direct_rdma_supported.is_some());
    assert!(second.gpu_direct_rdma_with_cuda_vmm_supported.is_some());
    assert!(second.device_free_memory_bytes.unwrap_or(0) > 0);
}
