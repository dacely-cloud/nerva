use nerva_core::types::memory::tier::MemoryTier;

pub(super) fn transfer_cost_ns(bytes: usize, old_tier: MemoryTier, new_tier: MemoryTier) -> u64 {
    if old_tier == new_tier {
        0
    } else {
        100 + bytes as u64
    }
}
