use crate::types::cost::CostEstimate;
use crate::types::dtype::DType;
use crate::types::error::{NervaError, Result};
use crate::types::id::{
    AllocationId, LayoutId, MemoryDomainId, ReplicaId, ResidentBlockId, UseDistance,
};
use crate::types::memory::{MemoryFabricKind, MemoryTier};
use crate::types::ownership::{AccessPolicy, CoherencePolicy, ExecutionOwner, MutationSemantics};
use crate::types::shape::BlockShape;

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
            shape: BlockShape::scalar(),
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
