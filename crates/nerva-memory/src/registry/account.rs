use nerva_core::types::memory::tier::MemoryTier;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TierAccount {
    pub tier: MemoryTier,
    pub capacity_bytes: usize,
    pub used_bytes: usize,
}

impl TierAccount {
    pub const fn remaining_bytes(self) -> usize {
        self.capacity_bytes.saturating_sub(self.used_bytes)
    }
}
