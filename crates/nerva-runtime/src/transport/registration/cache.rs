use std::collections::BTreeMap;

use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::block::resident::ResidentBlock;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::{ReplicaId, ResidentBlockId};
use nerva_core::types::memory::MemoryTier;

use crate::transport::registration::types::{
    TransportRegistration, TransportRegistrationBackend, TransportRegistrationKey,
    TransportRegistrationLookup,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportRegistrationCache {
    capacity: usize,
    next_generation: u64,
    entries: BTreeMap<TransportRegistrationKey, TransportRegistration>,
}

impl TransportRegistrationCache {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "transport registration cache capacity must be non-zero".to_string(),
            });
        }
        Ok(Self {
            capacity,
            next_generation: 1,
            entries: BTreeMap::new(),
        })
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn register(
        &mut self,
        block: &ResidentBlock,
        replica: ReplicaId,
        backend: TransportRegistrationBackend,
    ) -> Result<TransportRegistration> {
        validate_block_registerable(block, backend)?;
        if !block.residency.contains(replica) {
            return Err(NervaError::ResidencyViolation {
                block_id: block.id,
                reason: "transport registration replica is not resident".to_string(),
            });
        }
        if self.entries.len() == self.capacity {
            let Some(oldest) = self
                .entries
                .values()
                .min_by_key(|entry| entry.generation)
                .map(|entry| entry.key)
            else {
                return Err(NervaError::InvalidArgument {
                    reason: "transport registration cache eviction found no entry".to_string(),
                });
            };
            self.entries.remove(&oldest);
        }
        let key = TransportRegistrationKey {
            block_id: block.id,
            replica,
            backend,
        };
        let registration = TransportRegistration {
            key,
            address: block.address,
            tier: block.tier,
            bytes: block.bytes,
            registered_min_version: block.version,
            generation: self.next_generation,
        };
        self.next_generation = self.next_generation.saturating_add(1);
        self.entries.insert(key, registration);
        Ok(registration)
    }

    pub fn lookup(
        &self,
        block: &ResidentBlock,
        replica: ReplicaId,
        backend: TransportRegistrationBackend,
        required_version: u64,
    ) -> TransportRegistrationLookup {
        let key = TransportRegistrationKey {
            block_id: block.id,
            replica,
            backend,
        };
        let Some(registration) = self.entries.get(&key).copied() else {
            return TransportRegistrationLookup::Miss;
        };
        if registration.address != block.address
            || registration.tier != block.tier
            || registration.bytes != block.bytes
        {
            return TransportRegistrationLookup::StaleAddress(registration);
        }
        if block.version < required_version || registration.registered_min_version > block.version {
            return TransportRegistrationLookup::StaleVersion(registration);
        }
        TransportRegistrationLookup::Hit(registration)
    }

    pub fn revoke(&mut self, key: TransportRegistrationKey) -> Option<TransportRegistration> {
        self.entries.remove(&key)
    }

    pub fn revoke_block(&mut self, block_id: ResidentBlockId) -> Vec<TransportRegistration> {
        let keys = self
            .entries
            .keys()
            .copied()
            .filter(|key| key.block_id == block_id)
            .collect::<Vec<_>>();
        keys.into_iter()
            .filter_map(|key| self.entries.remove(&key))
            .collect()
    }

    pub fn revoke_all(&mut self) -> Vec<TransportRegistration> {
        let keys = self.entries.keys().copied().collect::<Vec<_>>();
        keys.into_iter()
            .filter_map(|key| self.entries.remove(&key))
            .collect()
    }
}

fn validate_block_registerable(
    block: &ResidentBlock,
    backend: TransportRegistrationBackend,
) -> Result<()> {
    if block.state != ResidencyState::Ready {
        return Err(NervaError::ResidencyViolation {
            block_id: block.id,
            reason: "transport registration requires a ready block".to_string(),
        });
    }
    match (backend, block.tier) {
        (TransportRegistrationBackend::RdmaPinnedHost, MemoryTier::PinnedDram)
        | (TransportRegistrationBackend::DpdkPinnedHost, MemoryTier::PinnedDram)
        | (TransportRegistrationBackend::RdmaGpuDirect, MemoryTier::Vram)
        | (TransportRegistrationBackend::DpdkGpu, MemoryTier::Vram) => Ok(()),
        _ => Err(NervaError::InvalidArgument {
            reason: format!(
                "backend {} cannot register {:?} blocks",
                backend.as_str(),
                block.tier
            ),
        }),
    }
}
