use std::collections::BTreeMap;

use nerva_core::types::block::resident::ResidentBlock;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;

use crate::registry::account::TierAccount;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockRegistry {
    pub(crate) next_id: u64,
    pub(crate) accounts: BTreeMap<MemoryTier, TierAccount>,
    pub(crate) blocks: BTreeMap<ResidentBlockId, ResidentBlock>,
}

impl BlockRegistry {
    pub fn new(accounts: impl IntoIterator<Item = (MemoryTier, usize)>) -> Self {
        let mut registry = Self {
            next_id: 1,
            accounts: BTreeMap::new(),
            blocks: BTreeMap::new(),
        };
        for (tier, capacity_bytes) in accounts {
            registry.accounts.insert(
                tier,
                TierAccount {
                    tier,
                    capacity_bytes,
                    occupied_bytes: 0,
                },
            );
        }
        registry
    }
}
