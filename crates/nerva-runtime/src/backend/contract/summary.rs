use nerva_core::types::backend::capabilities::{DeviceBackendCapabilities, DeviceBackendKind};
use nerva_core::types::backend::validation::BackendContractValidation;
use nerva_core::types::dtype::DType;
use nerva_core::types::memory::fabric::MemoryFabricKind;

use crate::backend::contract::json::{backend_kind_to_str, dtype_array};
use crate::backend::contract::status::BackendContractProbeStatus;
use crate::capabilities::json::{json_opt_string, json_opt_usize, memory_fabric_to_str};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeBackendContractSummary {
    pub status: BackendContractProbeStatus,
    pub backend: DeviceBackendKind,
    pub device_ordinal: i32,
    pub backend_name: Option<String>,
    pub architecture_major: Option<i32>,
    pub architecture_minor: Option<i32>,
    pub fabric: MemoryFabricKind,
    pub total_device_memory_bytes: Option<usize>,
    pub exact_dtypes: Vec<DType>,
    pub supports_device_allocations: bool,
    pub supports_pinned_host_allocations: bool,
    pub supports_streams: bool,
    pub supports_events: bool,
    pub supports_graph_capture: bool,
    pub supports_async_copies: bool,
    pub supports_device_sampling: bool,
    pub requested_device_bytes: usize,
    pub requested_pinned_bytes: usize,
    pub allocated_device_bytes: usize,
    pub allocated_pinned_bytes: usize,
    pub graph_replays: u64,
    pub graph_nodes: u64,
    pub sampler_tokens: u64,
    pub queue_ready: bool,
    pub event_ready: bool,
    pub graph_ready: bool,
    pub submission_id: Option<u64>,
    pub validation: BackendContractValidation,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl RuntimeBackendContractSummary {
    pub fn from_parts(
        status: BackendContractProbeStatus,
        capabilities: &DeviceBackendCapabilities,
        validation: BackendContractValidation,
    ) -> Self {
        Self {
            status,
            backend: capabilities.kind,
            device_ordinal: capabilities.device.0,
            backend_name: capabilities.name.clone(),
            architecture_major: capabilities.architecture.map(|arch| arch.major),
            architecture_minor: capabilities.architecture.map(|arch| arch.minor),
            fabric: capabilities.fabric,
            total_device_memory_bytes: capabilities.total_device_memory_bytes,
            exact_dtypes: capabilities.exact_dtypes.clone(),
            supports_device_allocations: capabilities.supports_device_allocations,
            supports_pinned_host_allocations: capabilities.supports_pinned_host_allocations,
            supports_streams: capabilities.supports_streams,
            supports_events: capabilities.supports_events,
            supports_graph_capture: capabilities.supports_graph_capture,
            supports_async_copies: capabilities.supports_async_copies,
            supports_device_sampling: capabilities.supports_device_sampling,
            requested_device_bytes: 0,
            requested_pinned_bytes: 0,
            allocated_device_bytes: 0,
            allocated_pinned_bytes: 0,
            graph_replays: 0,
            graph_nodes: 0,
            sampler_tokens: 0,
            queue_ready: false,
            event_ready: false,
            graph_ready: false,
            submission_id: None,
            validation,
            hot_path_allocations: 0,
            error: None,
        }
    }

    pub fn passed(&self) -> bool {
        self.status == BackendContractProbeStatus::Ok
            && self.queue_ready
            && self.event_ready
            && self.graph_ready
            && self.submission_id.is_some()
            && self.validation.passed()
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"status\":\"{}\",\"backend\":\"{}\",\"device_ordinal\":{},\"backend_name\":{},\"architecture_major\":{},\"architecture_minor\":{},\"fabric\":\"{}\",\"total_device_memory_bytes\":{},\"exact_dtypes\":{},\"supports_device_allocations\":{},\"supports_pinned_host_allocations\":{},\"supports_streams\":{},\"supports_events\":{},\"supports_graph_capture\":{},\"supports_async_copies\":{},\"supports_device_sampling\":{},\"requested_device_bytes\":{},\"requested_pinned_bytes\":{},\"allocated_device_bytes\":{},\"allocated_pinned_bytes\":{},\"graph_replays\":{},\"graph_nodes\":{},\"sampler_tokens\":{},\"queue_ready\":{},\"event_ready\":{},\"graph_ready\":{},\"submission_id\":{},\"bootstrap_decode_ready\":{},\"device_allocation_ready\":{},\"pinned_allocation_ready\":{},\"validation_queue_ready\":{},\"validation_graph_ready\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            self.status.as_str(),
            backend_kind_to_str(self.backend),
            self.device_ordinal,
            json_opt_string(self.backend_name.as_deref()),
            self.architecture_major
                .map_or_else(|| "null".to_string(), |value| value.to_string()),
            self.architecture_minor
                .map_or_else(|| "null".to_string(), |value| value.to_string()),
            memory_fabric_to_str(self.fabric),
            json_opt_usize(self.total_device_memory_bytes),
            dtype_array(&self.exact_dtypes),
            self.supports_device_allocations,
            self.supports_pinned_host_allocations,
            self.supports_streams,
            self.supports_events,
            self.supports_graph_capture,
            self.supports_async_copies,
            self.supports_device_sampling,
            self.requested_device_bytes,
            self.requested_pinned_bytes,
            self.allocated_device_bytes,
            self.allocated_pinned_bytes,
            self.graph_replays,
            self.graph_nodes,
            self.sampler_tokens,
            self.queue_ready,
            self.event_ready,
            self.graph_ready,
            self.submission_id
                .map_or_else(|| "null".to_string(), |value| value.to_string()),
            self.validation.bootstrap_decode_ready,
            self.validation.device_allocation_ready,
            self.validation.pinned_allocation_ready,
            self.validation.queue_ready,
            self.validation.graph_ready,
            self.hot_path_allocations,
            json_opt_string(self.error.as_deref()),
        )
    }
}
