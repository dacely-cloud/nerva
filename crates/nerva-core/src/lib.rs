#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!(
    "NERVA currently supports Linux only. Ubuntu x86_64 and aarch64 are the M0 host targets."
);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HostArch {
    X86_64,
    Aarch64,
    Other,
}

pub fn host_arch() -> HostArch {
    #[cfg(target_arch = "x86_64")]
    {
        HostArch::X86_64
    }
    #[cfg(target_arch = "aarch64")]
    {
        HostArch::Aarch64
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        HostArch::Other
    }
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct DeviceOrdinal(pub i32);

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ResidentBlockId(pub u64);

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct MemoryDomainId(pub u32);

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct AllocationId(pub u64);

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct ReplicaId(pub u64);

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct LayoutId(pub u32);

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct TransportDeviceId(pub u32);

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct UseDistance(pub u64);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MemoryFabricKind {
    DiscreteExplicit,
    UnifiedVirtualManaged,
    CoherentSharedPhysical,
    CxlCoherentFabric,
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MemoryTier {
    Vram,
    SharedHbmOrLpddr,
    PinnedDram,
    Dram,
    Cxl,
    Disk,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BlockKind {
    Weight,
    KvPage,
    Activation,
    Logits,
    TokenState,
    SamplerState,
    Workspace,
    Queue,
    TransportBuffer,
    Ledger,
    Metadata,
}

pub type ResidentBlockKind = BlockKind;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DType {
    U8,
    U16,
    U32,
    I32,
    F16,
    BF16,
    F32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockShape {
    dims: Vec<u64>,
}

impl BlockShape {
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    pub fn from_dims(dims: impl Into<Vec<u64>>) -> Result<Self> {
        let dims = dims.into();
        if dims.iter().any(|dim| *dim == 0) {
            return Err(NervaError::InvalidArgument {
                reason: "block shape dimensions must be non-zero".to_string(),
            });
        }
        Ok(Self { dims })
    }

    pub fn dims(&self) -> &[u64] {
        &self.dims
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ExecutionOwner {
    Cpu,
    Gpu(DeviceOrdinal),
    Nic(TransportDeviceId),
    SharedReadOnly,
    PhaseTransition,
    None,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CoherencePolicy {
    ExplicitVersioned,
    CoherentReadMostly,
    CoherentPhaseOwned,
    AtomicControlOnly,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AccessPolicy {
    CpuOnly,
    GpuOnly,
    NicOnly,
    CpuGpuReadOnly,
    CpuThenGpu,
    GpuThenCpu,
    GpuThenNic,
    NicThenGpu,
    PhaseOwned,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MutationSemantics {
    Immutable,
    AppendOnly,
    SingleWriter,
    Ephemeral,
    AtomicControl,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResidencyState {
    Unmapped,
    Allocated,
    Prefetching,
    Ready,
    InUse,
    Draining,
    Evicting,
    Invalid,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Lifetime {
    Static,
    Request,
    Token,
    Scratch,
    External,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Hotness {
    Cold,
    Warm,
    Hot,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CostSource {
    Unknown,
    Estimated,
    Measured,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CostEstimate {
    pub nanos: Option<u64>,
    pub source: CostSource,
}

impl CostEstimate {
    pub const fn unknown() -> Self {
        Self {
            nanos: None,
            source: CostSource::Unknown,
        }
    }

    pub const fn estimated_nanos(nanos: u64) -> Self {
        Self {
            nanos: Some(nanos),
            source: CostSource::Estimated,
        }
    }

    pub const fn measured_nanos(nanos: u64) -> Self {
        Self {
            nanos: Some(nanos),
            source: CostSource::Measured,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GlobalBlockAddress {
    pub domain: MemoryDomainId,
    pub allocation: AllocationId,
    pub offset: u64,
}

impl GlobalBlockAddress {
    pub const fn unmapped() -> Self {
        Self {
            domain: MemoryDomainId(0),
            allocation: AllocationId(0),
            offset: 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidencySet {
    replicas: Vec<ReplicaId>,
}

impl ResidencySet {
    pub fn empty() -> Self {
        Self {
            replicas: Vec::new(),
        }
    }

    pub fn single(replica: ReplicaId) -> Self {
        Self {
            replicas: vec![replica],
        }
    }

    pub fn contains(&self, replica: ReplicaId) -> bool {
        self.replicas.contains(&replica)
    }

    pub fn add(&mut self, replica: ReplicaId) {
        if !self.contains(replica) {
            self.replicas.push(replica);
        }
    }

    pub fn replicas(&self) -> &[ReplicaId] {
        &self.replicas
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct BlockFlags {
    bits: u32,
}

impl BlockFlags {
    pub const PREFETCHABLE: u32 = 1 << 0;
    pub const EVICTABLE: u32 = 1 << 1;
    pub const TRANSPORT_REGISTERED: u32 = 1 << 2;

    pub const fn empty() -> Self {
        Self { bits: 0 }
    }

    pub const fn from_bits(bits: u32) -> Self {
        Self { bits }
    }

    pub const fn bits(self) -> u32 {
        self.bits
    }

    pub const fn contains(self, flag: u32) -> bool {
        (self.bits & flag) == flag
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentBlock {
    pub id: ResidentBlockId,
    pub kind: BlockKind,
    pub bytes: usize,
    pub dtype: DType,
    pub shape: BlockShape,
    pub layout: LayoutId,
    pub address: GlobalBlockAddress,
    pub residency: ResidencySet,
    pub authoritative_copy: ReplicaId,
    pub version: u64,
    pub memory_domain: MemoryDomainId,
    pub fabric: MemoryFabricKind,
    pub owner: ExecutionOwner,
    pub coherence: CoherencePolicy,
    pub access: AccessPolicy,
    pub semantics: MutationSemantics,
    pub lifetime: Lifetime,
    pub hotness: Hotness,
    pub next_use: Option<UseDistance>,
    pub reuse_distance: Option<UseDistance>,
    pub read_cost: CostEstimate,
    pub write_cost: CostEstimate,
    pub move_cost: CostEstimate,
    pub compute_near_data_cost: CostEstimate,
    pub state: ResidencyState,
    pub flags: BlockFlags,
    pub tier: MemoryTier,
}

impl ResidentBlock {
    pub fn new(id: ResidentBlockId, kind: BlockKind, tier: MemoryTier, bytes: usize) -> Self {
        let domain = MemoryDomainId::for_tier(tier);
        let replica = ReplicaId(id.0);
        Self {
            id,
            kind,
            bytes,
            dtype: DType::U8,
            shape: BlockShape { dims: Vec::new() },
            layout: LayoutId(0),
            address: GlobalBlockAddress {
                domain,
                allocation: AllocationId(id.0),
                offset: 0,
            },
            residency: ResidencySet {
                replicas: vec![replica],
            },
            authoritative_copy: replica,
            version: 0,
            memory_domain: domain,
            fabric: MemoryFabricKind::DiscreteExplicit,
            owner: ExecutionOwner::None,
            coherence: CoherencePolicy::ExplicitVersioned,
            access: AccessPolicy::PhaseOwned,
            semantics: default_mutation_semantics(kind),
            lifetime: default_lifetime(kind),
            hotness: default_hotness(tier),
            next_use: None,
            reuse_distance: None,
            read_cost: CostEstimate::unknown(),
            write_cost: CostEstimate::unknown(),
            move_cost: CostEstimate::unknown(),
            compute_near_data_cost: CostEstimate::unknown(),
            state: ResidencyState::Allocated,
            flags: BlockFlags::empty(),
            tier,
        }
    }

    pub fn with_shape(mut self, dtype: DType, shape: BlockShape, layout: LayoutId) -> Self {
        self.dtype = dtype;
        self.shape = shape;
        self.layout = layout;
        self
    }

    pub fn with_address(mut self, address: GlobalBlockAddress) -> Self {
        self.memory_domain = address.domain;
        self.address = address;
        self
    }

    pub fn mark_ready(&mut self) {
        self.state = ResidencyState::Ready;
    }

    pub fn mark_in_use(&mut self) -> Result<()> {
        if self.state != ResidencyState::Ready {
            return Err(NervaError::ResidencyViolation {
                block_id: self.id,
                reason: "block must be ready before use".to_string(),
            });
        }
        self.state = ResidencyState::InUse;
        Ok(())
    }

    pub fn publish(&mut self, owner: ExecutionOwner) -> u64 {
        self.owner = owner;
        self.version = self.version.saturating_add(1);
        self.state = ResidencyState::Ready;
        self.version
    }

    pub fn ready_for(&self, replica: ReplicaId, min_version: u64) -> bool {
        self.state == ResidencyState::Ready
            && self.residency.contains(replica)
            && self.version >= min_version
    }

    pub fn require_ready(&self, replica: ReplicaId, min_version: u64) -> Result<()> {
        if self.state != ResidencyState::Ready {
            return Err(NervaError::ResidencyViolation {
                block_id: self.id,
                reason: "required block replica is not ready".to_string(),
            });
        }
        if !self.residency.contains(replica) {
            return Err(NervaError::ResidencyViolation {
                block_id: self.id,
                reason: "required block replica is not resident".to_string(),
            });
        }
        if self.version < min_version {
            return Err(NervaError::ResidencyViolation {
                block_id: self.id,
                reason: "required block version is stale".to_string(),
            });
        }
        Ok(())
    }
}

impl MemoryDomainId {
    pub const CPU_DRAM: Self = Self(1);
    pub const GPU_VRAM: Self = Self(2);
    pub const PINNED_DRAM: Self = Self(3);
    pub const SHARED_HBM_OR_LPDDR: Self = Self(4);
    pub const CXL: Self = Self(5);
    pub const DISK: Self = Self(6);

    pub const fn for_tier(tier: MemoryTier) -> Self {
        match tier {
            MemoryTier::Vram => Self::GPU_VRAM,
            MemoryTier::SharedHbmOrLpddr => Self::SHARED_HBM_OR_LPDDR,
            MemoryTier::PinnedDram => Self::PINNED_DRAM,
            MemoryTier::Dram => Self::CPU_DRAM,
            MemoryTier::Cxl => Self::CXL,
            MemoryTier::Disk => Self::DISK,
        }
    }
}

const fn default_mutation_semantics(kind: BlockKind) -> MutationSemantics {
    match kind {
        BlockKind::Weight => MutationSemantics::Immutable,
        BlockKind::KvPage => MutationSemantics::AppendOnly,
        BlockKind::Activation | BlockKind::Logits | BlockKind::Workspace => {
            MutationSemantics::Ephemeral
        }
        BlockKind::TokenState
        | BlockKind::SamplerState
        | BlockKind::Queue
        | BlockKind::Ledger
        | BlockKind::Metadata
        | BlockKind::TransportBuffer => MutationSemantics::SingleWriter,
    }
}

const fn default_lifetime(kind: BlockKind) -> Lifetime {
    match kind {
        BlockKind::Weight => Lifetime::Static,
        BlockKind::KvPage
        | BlockKind::TokenState
        | BlockKind::SamplerState
        | BlockKind::Queue
        | BlockKind::Ledger
        | BlockKind::Metadata
        | BlockKind::TransportBuffer => Lifetime::Request,
        BlockKind::Activation | BlockKind::Logits => Lifetime::Token,
        BlockKind::Workspace => Lifetime::Scratch,
    }
}

const fn default_hotness(tier: MemoryTier) -> Hotness {
    match tier {
        MemoryTier::Vram | MemoryTier::SharedHbmOrLpddr => Hotness::Hot,
        MemoryTier::PinnedDram | MemoryTier::Dram | MemoryTier::Cxl => Hotness::Warm,
        MemoryTier::Disk => Hotness::Cold,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NervaError {
    UnsupportedHost {
        arch: HostArch,
    },
    BackendUnavailable {
        backend: &'static str,
        reason: String,
    },
    AllocationFailed {
        bytes: usize,
        reason: String,
    },
    InvalidArgument {
        reason: String,
    },
    ResidencyViolation {
        block_id: ResidentBlockId,
        reason: String,
    },
}

pub type Result<T> = std::result::Result<T, NervaError>;

pub fn ensure_supported_linux_host() -> Result<()> {
    match host_arch() {
        HostArch::X86_64 | HostArch::Aarch64 => Ok(()),
        arch => Err(NervaError::UnsupportedHost { arch }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_host_gate_accepts_build_host_or_reports_other() {
        let result = ensure_supported_linux_host();
        if matches!(host_arch(), HostArch::Other) {
            assert!(matches!(result, Err(NervaError::UnsupportedHost { .. })));
        } else {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn resident_block_carries_identity_and_tier() {
        let block = ResidentBlock::new(
            ResidentBlockId(7),
            BlockKind::KvPage,
            MemoryTier::Vram,
            4096,
        );
        assert_eq!(block.id, ResidentBlockId(7));
        assert_eq!(block.tier, MemoryTier::Vram);
        assert_eq!(block.memory_domain, MemoryDomainId::GPU_VRAM);
        assert_eq!(block.semantics, MutationSemantics::AppendOnly);
    }

    #[test]
    fn ready_check_rejects_stale_or_absent_replicas() {
        let mut block = ResidentBlock::new(
            ResidentBlockId(11),
            BlockKind::TokenState,
            MemoryTier::Vram,
            64,
        );
        let replica = block.authoritative_copy;

        assert!(block.require_ready(replica, 0).is_err());
        block.mark_ready();
        assert!(block.require_ready(ReplicaId(999), 0).is_err());
        assert!(block.require_ready(replica, 1).is_err());

        let version = block.publish(ExecutionOwner::Gpu(DeviceOrdinal(0)));
        assert_eq!(version, 1);
        assert!(block.require_ready(replica, 1).is_ok());
    }

    #[test]
    fn block_shape_rejects_zero_dimensions() {
        assert_eq!(BlockShape::from_dims([2, 4]).unwrap().dims(), &[2, 4]);
        assert!(BlockShape::from_dims([2, 0]).is_err());
    }
}
