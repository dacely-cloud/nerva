use nerva_core::types::id::ResidentBlockId;
use nerva_core::types::memory::MemoryTier;

pub(crate) struct ResidentMatvecShard<'a> {
    pub(crate) block_id: ResidentBlockId,
    pub(crate) tier: MemoryTier,
    pub(crate) row_start: usize,
    pub(crate) row_end: usize,
    pub(crate) weights: &'a [f32],
}
