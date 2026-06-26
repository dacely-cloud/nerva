use crate::types::dtype::DType;
use crate::types::id::DeviceOrdinal;
use crate::types::memory::MemoryFabricKind;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DeviceBackendKind {
    Cpu,
    Cuda,
    Hip,
}

impl DeviceBackendKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Hip => "hip",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BackendArchitecture {
    pub major: i32,
    pub minor: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceBackendCapabilities {
    pub kind: DeviceBackendKind,
    pub device: DeviceOrdinal,
    pub name: Option<String>,
    pub architecture: Option<BackendArchitecture>,
    pub fabric: MemoryFabricKind,
    pub total_device_memory_bytes: Option<usize>,
    pub supports_device_allocations: bool,
    pub supports_pinned_host_allocations: bool,
    pub supports_streams: bool,
    pub supports_events: bool,
    pub supports_graph_capture: bool,
    pub supports_async_copies: bool,
    pub supports_device_sampling: bool,
    pub exact_dtypes: Vec<DType>,
}

impl DeviceBackendCapabilities {
    pub fn supports_exact_dtype(&self, dtype: DType) -> bool {
        self.exact_dtypes.contains(&dtype)
    }

    pub fn supports_bootstrap_decode_contract(&self) -> bool {
        self.supports_device_allocations
            && self.supports_pinned_host_allocations
            && self.supports_streams
            && self.supports_events
            && self.supports_graph_capture
            && self.supports_async_copies
            && self.supports_device_sampling
            && self.supports_exact_dtype(DType::F16)
    }
}
