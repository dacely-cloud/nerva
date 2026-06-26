use nerva_core::types::error::{NervaError, Result};
use nerva_cuda::backend::summary::CudaBackendContractSummary;
use nerva_cuda::graph::summary::CudaSyntheticGraphSummary;
use nerva_cuda::sampler::summary::CudaGreedySamplerSummary;
use nerva_cuda::smoke::status::SmokeStatus;

pub(super) fn validate_cuda_probe_inputs(
    backend: &CudaBackendContractSummary,
    graph: &CudaSyntheticGraphSummary,
    sampler: &CudaGreedySamplerSummary,
) -> Result<()> {
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
    Ok(())
}
