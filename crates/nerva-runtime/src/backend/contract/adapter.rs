use nerva_core::types::backend::capabilities::{
    BackendArchitecture, DeviceBackendCapabilities, DeviceBackendKind,
};
use nerva_core::types::backend::contract::DeviceBackend;
use nerva_core::types::backend::operation::{
    BackendAllocationContract, BackendDeviceHandle, BackendEventContract, BackendGraphExecContract,
    BackendQueueContract, BackendSubmissionId, BackendTransactionDescriptor,
};
use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::memory::fabric::MemoryFabricKind;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_cuda::backend::summary::CudaBackendContractSummary;
use nerva_cuda::graph::summary::CudaSyntheticGraphSummary;
use nerva_cuda::sampler::summary::CudaGreedySamplerSummary;
use nerva_cuda::smoke::status::SmokeStatus;

#[derive(Clone, Debug)]
pub struct CudaBackendContractAdapter {
    capabilities: DeviceBackendCapabilities,
    proven_device_bytes: usize,
    proven_pinned_bytes: usize,
    next_submission_id: u64,
}

impl CudaBackendContractAdapter {
    pub fn from_probe(
        backend: &CudaBackendContractSummary,
        graph: &CudaSyntheticGraphSummary,
        sampler: &CudaGreedySamplerSummary,
    ) -> Result<Self> {
        if !backend.passed() {
            return Err(NervaError::BackendUnavailable {
                backend: "cuda",
                reason: backend
                    .error
                    .clone()
                    .unwrap_or_else(|| "CUDA allocation and queue contract failed".to_string()),
            });
        }
        if graph.status != SmokeStatus::Ok || graph.hot_path_allocations != 0 {
            return Err(NervaError::BackendUnavailable {
                backend: "cuda",
                reason: graph
                    .error
                    .clone()
                    .unwrap_or_else(|| "CUDA graph contract failed".to_string()),
            });
        }
        if sampler.status != SmokeStatus::Ok || sampler.hot_path_allocations != 0 {
            return Err(NervaError::BackendUnavailable {
                backend: "cuda",
                reason: sampler
                    .error
                    .clone()
                    .unwrap_or_else(|| "CUDA device sampling contract failed".to_string()),
            });
        }

        Ok(Self {
            capabilities: cuda_capabilities_from_probe(backend, true, true),
            proven_device_bytes: backend.allocated_device_bytes,
            proven_pinned_bytes: backend.allocated_pinned_bytes,
            next_submission_id: 1,
        })
    }
}

impl DeviceBackend for CudaBackendContractAdapter {
    type Device = BackendDeviceHandle;
    type Queue = BackendQueueContract;
    type Event = BackendEventContract;
    type GraphExec = BackendGraphExecContract;
    type DeviceAllocation = BackendAllocationContract;
    type PinnedAllocation = BackendAllocationContract;

    fn capabilities(&self) -> &DeviceBackendCapabilities {
        &self.capabilities
    }

    fn create_device(&self, id: DeviceOrdinal) -> Result<Self::Device> {
        if id != self.capabilities.device {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "backend device ordinal {} does not match requested ordinal {}",
                    self.capabilities.device.0, id.0
                ),
            });
        }
        Ok(BackendDeviceHandle { device: id })
    }

    fn create_queue(&self, device: &Self::Device) -> Result<Self::Queue> {
        if !self.capabilities.supports_streams {
            return Err(NervaError::BackendUnavailable {
                backend: "cuda",
                reason: "CUDA stream support was not proven".to_string(),
            });
        }
        Ok(BackendQueueContract {
            device: device.device,
            bounded: true,
            stream_ordered: true,
            preallocated: true,
        })
    }

    fn create_event(&self, device: &Self::Device) -> Result<Self::Event> {
        if !self.capabilities.supports_events {
            return Err(NervaError::BackendUnavailable {
                backend: "cuda",
                reason: "CUDA event support was not proven".to_string(),
            });
        }
        Ok(BackendEventContract {
            device: device.device,
            timing_enabled: false,
            preallocated: true,
        })
    }

    fn allocate_device(
        &self,
        _device: &Self::Device,
        bytes: usize,
        alignment: usize,
    ) -> Result<Self::DeviceAllocation> {
        if !self.capabilities.supports_device_allocations {
            return Err(NervaError::BackendUnavailable {
                backend: "cuda",
                reason: "CUDA device allocation support was not proven".to_string(),
            });
        }
        if bytes == 0 || bytes > self.proven_device_bytes || alignment == 0 {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: format!(
                    "requested device allocation exceeds proven CUDA allocation bytes {} or has invalid alignment {}",
                    self.proven_device_bytes, alignment
                ),
            });
        }
        Ok(BackendAllocationContract {
            tier: MemoryTier::Vram,
            bytes,
            alignment,
            preallocated: true,
        })
    }

    fn allocate_pinned(&self, bytes: usize, alignment: usize) -> Result<Self::PinnedAllocation> {
        if !self.capabilities.supports_pinned_host_allocations {
            return Err(NervaError::BackendUnavailable {
                backend: "cuda",
                reason: "CUDA pinned-host allocation support was not proven".to_string(),
            });
        }
        if bytes == 0 || bytes > self.proven_pinned_bytes || alignment == 0 {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: format!(
                    "requested pinned allocation exceeds proven CUDA allocation bytes {} or has invalid alignment {}",
                    self.proven_pinned_bytes, alignment
                ),
            });
        }
        Ok(BackendAllocationContract {
            tier: MemoryTier::PinnedDram,
            bytes,
            alignment,
            preallocated: true,
        })
    }

    fn capture(&self, transaction: &BackendTransactionDescriptor) -> Result<Self::GraphExec> {
        if !self.capabilities.supports_graph_capture {
            return Err(NervaError::BackendUnavailable {
                backend: "cuda",
                reason: "CUDA graph capture support was not proven".to_string(),
            });
        }
        if !transaction.graph_capturable {
            return Err(NervaError::InvalidArgument {
                reason: "transaction is not graph-capturable".to_string(),
            });
        }
        Ok(BackendGraphExecContract {
            transaction: *transaction,
            replayable: true,
        })
    }

    fn submit(&mut self, executable: &Self::GraphExec) -> Result<BackendSubmissionId> {
        if !executable.replayable {
            return Err(NervaError::InvalidArgument {
                reason: "graph executable is not replayable".to_string(),
            });
        }
        let submission = BackendSubmissionId(self.next_submission_id);
        self.next_submission_id = self.next_submission_id.saturating_add(1);
        Ok(submission)
    }
}

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
