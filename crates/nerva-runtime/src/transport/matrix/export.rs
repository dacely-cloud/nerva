use crate::capabilities::snapshot::{CapabilitySnapshot, CapabilityState};
use crate::transport::matrix::types::TransportMatrixRequestedPath;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct MatrixExportEvidence {
    pub gpu_memory_export_verified: bool,
    pub cuda_vmm_posix_fd_export_verified: bool,
    pub gpu_direct_rdma_verified: bool,
    pub gpu_export_without_nic_direct: bool,
}

pub(crate) fn transport_matrix_export_evidence(
    requested_path: TransportMatrixRequestedPath,
    capabilities: &CapabilitySnapshot,
) -> MatrixExportEvidence {
    let gpu_memory_export_verified = capabilities.dma_buf_export
        == CapabilityState::SupportedAndVerified
        || capabilities.cuda_vmm_posix_fd_export_verified == Some(true);
    let cuda_vmm_posix_fd_export_verified =
        capabilities.cuda_vmm_posix_fd_export_verified == Some(true);
    let gpu_direct_rdma_verified = capabilities.gpu_direct_rdma
        == CapabilityState::SupportedAndVerified
        || capabilities.cuda_gpu_direct_rdma_supported == Some(true)
        || capabilities.cuda_gpu_direct_rdma_with_vmm_supported == Some(true);
    let direct_request = requested_path == TransportMatrixRequestedPath::GpuDirectRdma;

    MatrixExportEvidence {
        gpu_memory_export_verified: direct_request && gpu_memory_export_verified,
        cuda_vmm_posix_fd_export_verified: direct_request && cuda_vmm_posix_fd_export_verified,
        gpu_direct_rdma_verified: direct_request && gpu_direct_rdma_verified,
        gpu_export_without_nic_direct: direct_request
            && gpu_memory_export_verified
            && !gpu_direct_rdma_verified,
    }
}
