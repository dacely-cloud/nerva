use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;

use crate::registry::account::TierAccount;
use crate::registry::table::registry::BlockRegistry;

impl BlockRegistry {
    pub fn account(&self, tier: MemoryTier) -> Option<TierAccount> {
        self.accounts.get(&tier).copied()
    }

    pub fn used_bytes(&self, tier: MemoryTier) -> usize {
        self.account(tier).map_or(0, |account| account.used_bytes)
    }

    pub fn remaining_bytes(&self, tier: MemoryTier) -> Option<usize> {
        self.account(tier).map(|account| account.remaining_bytes())
    }

    pub(crate) fn reserve_tier(&mut self, tier: MemoryTier, bytes: usize) -> Result<()> {
        let account = self
            .accounts
            .get_mut(&tier)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("memory tier {tier:?} is not configured"),
            })?;
        let new_used =
            account
                .used_bytes
                .checked_add(bytes)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes,
                    reason: "tier accounting overflow".to_string(),
                })?;
        if new_used > account.capacity_bytes {
            return Err(NervaError::AllocationFailed {
                bytes,
                reason: format!("memory tier {tier:?} exhausted"),
            });
        }
        account.used_bytes = new_used;
        Ok(())
    }

    pub(crate) fn release_tier(&mut self, tier: MemoryTier, bytes: usize) {
        if let Some(account) = self.accounts.get_mut(&tier) {
            account.used_bytes = account.used_bytes.saturating_sub(bytes);
        }
    }
}
