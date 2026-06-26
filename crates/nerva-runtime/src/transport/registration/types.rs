use nerva_core::types::block::address::GlobalBlockAddress;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::replica::ReplicaId;

use nerva_core::types::memory::tier::MemoryTier;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum TransportRegistrationBackend {
    RdmaGpuDirect,
    RdmaPinnedHost,
    DpdkGpu,
    DpdkPinnedHost,
}

impl TransportRegistrationBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RdmaGpuDirect => "rdma_gpu_direct",
            Self::RdmaPinnedHost => "rdma_pinned_host",
            Self::DpdkGpu => "dpdk_gpu",
            Self::DpdkPinnedHost => "dpdk_pinned_host",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct TransportRegistrationKey {
    pub block_id: ResidentBlockId,
    pub replica: ReplicaId,
    pub backend: TransportRegistrationBackend,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportRegistration {
    pub key: TransportRegistrationKey,
    pub address: GlobalBlockAddress,
    pub tier: MemoryTier,
    pub bytes: usize,
    pub registered_min_version: u64,
    pub generation: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransportRegistrationLookup {
    Hit(TransportRegistration),
    Miss,
    StaleAddress(TransportRegistration),
    StaleVersion(TransportRegistration),
}
