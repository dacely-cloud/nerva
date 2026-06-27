use crate::json::{json_opt_i32, json_opt_str, json_opt_u32, json_opt_usize};
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaBackendContractSummary {
    pub status: SmokeStatus,
    pub gpu_name: Option<String>,
    pub driver_version: Option<i32>,
    pub runtime_version: Option<i32>,
    pub compute_capability_major: Option<i32>,
    pub compute_capability_minor: Option<i32>,
    pub device_total_memory_bytes: Option<usize>,
    pub device_free_memory_bytes: Option<usize>,
    pub pci_bus_id: Option<String>,
    pub device_count: i32,
    pub device_ordinal: i32,
    pub requested_device_bytes: usize,
    pub requested_pinned_bytes: usize,
    pub allocated_device_bytes: usize,
    pub allocated_pinned_bytes: usize,
    pub stream_creations: u64,
    pub stream_destroys: u64,
    pub event_creations: u64,
    pub event_destroys: u64,
    pub device_allocations: u64,
    pub device_frees: u64,
    pub pinned_allocations: u64,
    pub pinned_frees: u64,
    pub memset_bytes: u64,
    pub d2h_bytes: u64,
    pub sync_calls: u64,
    pub observed_word: Option<u32>,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaBackendContractSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"gpu_name\":{},\"driver_version\":{},\"runtime_version\":{},\"compute_capability_major\":{},\"compute_capability_minor\":{},\"device_total_memory_bytes\":{},\"device_free_memory_bytes\":{},\"pci_bus_id\":{},\"device_count\":{},\"device_ordinal\":{},\"requested_device_bytes\":{},\"requested_pinned_bytes\":{},\"allocated_device_bytes\":{},\"allocated_pinned_bytes\":{},\"stream_creations\":{},\"stream_destroys\":{},\"event_creations\":{},\"event_destroys\":{},\"device_allocations\":{},\"device_frees\":{},\"pinned_allocations\":{},\"pinned_frees\":{},\"memset_bytes\":{},\"D2H_bytes\":{},\"sync_calls\":{},\"observed_word\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            json_opt_str(self.gpu_name.as_deref()),
            json_opt_i32(self.driver_version),
            json_opt_i32(self.runtime_version),
            json_opt_i32(self.compute_capability_major),
            json_opt_i32(self.compute_capability_minor),
            json_opt_usize(self.device_total_memory_bytes),
            json_opt_usize(self.device_free_memory_bytes),
            json_opt_str(self.pci_bus_id.as_deref()),
            self.device_count,
            self.device_ordinal,
            self.requested_device_bytes,
            self.requested_pinned_bytes,
            self.allocated_device_bytes,
            self.allocated_pinned_bytes,
            self.stream_creations,
            self.stream_destroys,
            self.event_creations,
            self.event_destroys,
            self.device_allocations,
            self.device_frees,
            self.pinned_allocations,
            self.pinned_frees,
            self.memset_bytes,
            self.d2h_bytes,
            self.sync_calls,
            json_opt_u32(self.observed_word),
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
    }

    pub fn passed(&self) -> bool {
        self.status == SmokeStatus::Ok
            && self.stream_creations == 1
            && self.stream_destroys == 1
            && self.event_creations == 1
            && self.event_destroys == 1
            && self.device_allocations == 1
            && self.device_frees == 1
            && self.pinned_allocations == 1
            && self.pinned_frees == 1
            && self.allocated_device_bytes == self.requested_device_bytes
            && self.allocated_pinned_bytes == self.requested_pinned_bytes
            && self.memset_bytes == self.requested_device_bytes as u64
            && self.d2h_bytes == core::mem::size_of::<u32>() as u64
            && self.sync_calls == 1
            && self.observed_word == Some(0x5a5a_5a5a)
            && self.hot_path_allocations == 0
    }
}
