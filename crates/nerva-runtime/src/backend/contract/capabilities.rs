use nerva_core::types::backend::capabilities::{
    BackendArchitecture, DeviceBackendCapabilities, DeviceBackendKind,
};
use nerva_core::types::dtype::DType;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::memory::fabric::MemoryFabricKind;
use nerva_cuda::backend::summary::CudaBackendContractSummary;

pub fn cuda_capabilities_from_probe(
    backend: &CudaBackendContractSummary,
    supports_graph_capture: bool,
    supports_device_sampling: bool,
) -> DeviceBackendCapabilities {
    DeviceBackendCapabilities {
        kind: DeviceBackendKind::Cuda,
        device: DeviceOrdinal(backend.device_ordinal.max(0)),
        name: backend.gpu_name.clone(),
        architecture: match (
            backend.compute_capability_major,
            backend.compute_capability_minor,
        ) {
            (Some(major), Some(minor)) => Some(BackendArchitecture { major, minor }),
            _ => None,
        },
        fabric: MemoryFabricKind::DiscreteExplicit,
        total_device_memory_bytes: backend.device_total_memory_bytes,
        supports_device_allocations: backend.device_allocations > 0 && backend.device_frees > 0,
        supports_pinned_host_allocations: backend.pinned_allocations > 0
            && backend.pinned_frees > 0,
        supports_streams: backend.stream_creations > 0 && backend.stream_destroys > 0,
        supports_events: backend.event_creations > 0 && backend.event_destroys > 0,
        supports_graph_capture,
        supports_async_copies: backend.d2h_bytes > 0,
        supports_device_sampling,
        exact_dtypes: vec![DType::F16],
    }
}
