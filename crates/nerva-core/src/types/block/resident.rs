use crate::types::cost::estimate::CostEstimate;
use crate::types::dtype::DType;
use crate::types::error::{NervaError, Result};
use crate::types::id::allocation::AllocationId;
use crate::types::id::block::ResidentBlockId;
use crate::types::id::layout::LayoutId;
use crate::types::id::memory::MemoryDomainId;
use crate::types::id::replica::ReplicaId;
use crate::types::id::use_distance::UseDistance;

use crate::types::memory::fabric::MemoryFabricKind;
use crate::types::memory::tier::MemoryTier;
use crate::types::ownership::access::AccessPolicy;
use crate::types::ownership::coherence::CoherencePolicy;
use crate::types::ownership::mutation::MutationSemantics;
use crate::types::ownership::owner::ExecutionOwner;

use crate::types::shape::BlockShape;

use super::address::GlobalBlockAddress;
use super::defaults::{default_hotness, default_lifetime, default_mutation_semantics};
use super::flags::BlockFlags;
use super::hotness::Hotness;
use super::kind::BlockKind;
use super::lifetime::Lifetime;
use super::residency::{ResidencySet, ResidencyState};

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
            residency: ResidencySet::single(replica),
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
