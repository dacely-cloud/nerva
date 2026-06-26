use nerva_core::types::block::address::GlobalBlockAddress;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::allocation::AllocationId;
use nerva_core::types::memory::tier::MemoryTier;

use crate::arena::kind::{AllocationPhase, ArenaKind};
use crate::arena::region::{ArenaCheckpoint, ArenaRegion};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticArena {
    kind: ArenaKind,
    allocation: AllocationId,
    capacity_bytes: usize,
    used_bytes: usize,
}

impl StaticArena {
    pub const fn new(kind: ArenaKind, allocation: AllocationId, capacity_bytes: usize) -> Self {
        Self {
            kind,
            allocation,
            capacity_bytes,
            used_bytes: 0,
        }
    }

    pub const fn kind(&self) -> ArenaKind {
        self.kind
    }

    pub const fn tier(&self) -> MemoryTier {
        self.kind.tier()
    }

    pub const fn domain(&self) -> nerva_core::types::id::memory::MemoryDomainId {
        self.kind.domain()
    }

    pub const fn allocation(&self) -> AllocationId {
        self.allocation
    }

    pub const fn capacity(&self) -> usize {
        self.capacity_bytes
    }

    pub const fn used(&self) -> usize {
        self.used_bytes
    }

    pub const fn remaining(&self) -> usize {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }

    pub const fn checkpoint(&self) -> ArenaCheckpoint {
        ArenaCheckpoint {
            used: self.used_bytes,
        }
    }

    pub fn restore(&mut self, checkpoint: ArenaCheckpoint) -> Result<()> {
        if checkpoint.used > self.used_bytes {
            return Err(NervaError::InvalidArgument {
                reason: "arena checkpoint is ahead of current usage".to_string(),
            });
        }
        self.used_bytes = checkpoint.used;
        Ok(())
    }

    pub fn reserve(
        &mut self,
        name: &'static str,
        bytes: usize,
        align: usize,
        phase: AllocationPhase,
    ) -> Result<ArenaRegion> {
        if phase == AllocationPhase::HotPath {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: "static arena allocation attempted during hot path".to_string(),
            });
        }
        let align = align.max(1);
        let offset = self.used_bytes.next_multiple_of(align);
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes,
                reason: "static arena offset overflow".to_string(),
            })?;
        if end > self.capacity_bytes {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: format!("static {:?} arena exhausted", self.kind),
            });
        }
        self.used_bytes = end;
        Ok(ArenaRegion {
            name,
            kind: self.kind,
            tier: self.tier(),
            address: GlobalBlockAddress {
                domain: self.domain(),
                allocation: self.allocation,
                offset: offset as u64,
            },
            offset,
            bytes,
            align,
        })
    }
}
