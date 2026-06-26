use crate::json::{json_opt_bool, json_opt_i32, json_opt_str, json_opt_u32, json_opt_usize};
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaSmokeSummary {
    pub status: SmokeStatus,
    pub gpu_name: Option<String>,
    pub driver_version: Option<i32>,
    pub runtime_version: Option<i32>,
    pub compute_capability_major: Option<i32>,
    pub compute_capability_minor: Option<i32>,
    pub posix_fd_handle_supported: Option<bool>,
    pub gpu_direct_rdma_supported: Option<bool>,
    pub gpu_direct_rdma_with_cuda_vmm_supported: Option<bool>,
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
            "{{\"status\":\"{}\",\"gpu_name\":{},\"driver_version\":{},\"runtime_version\":{},\"compute_capability_major\":{},\"compute_capability_minor\":{},\"posix_fd_handle_supported\":{},\"gpu_direct_rdma_supported\":{},\"gpu_direct_rdma_with_cuda_vmm_supported\":{},\"device_total_memory_bytes\":{},\"pci_bus_id\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"kernel_value\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            json_opt_str(self.gpu_name.as_deref()),
            json_opt_i32(self.driver_version),
            json_opt_i32(self.runtime_version),
            json_opt_i32(self.compute_capability_major),
            json_opt_i32(self.compute_capability_minor),
            json_opt_bool(self.posix_fd_handle_supported),
            json_opt_bool(self.gpu_direct_rdma_supported),
            json_opt_bool(self.gpu_direct_rdma_with_cuda_vmm_supported),
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
            posix_fd_handle_supported: None,
            gpu_direct_rdma_supported: None,
            gpu_direct_rdma_with_cuda_vmm_supported: None,
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
            posix_fd_handle_supported: None,
            gpu_direct_rdma_supported: None,
            gpu_direct_rdma_with_cuda_vmm_supported: None,
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
