use nerva_core::types::memory::tier::MemoryTier;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KvAttentionBlock<'a> {
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub token_count: usize,
    pub tier: MemoryTier,
}

impl<'a> KvAttentionBlock<'a> {
    pub const fn new(
        keys: &'a [f32],
        values: &'a [f32],
        token_count: usize,
        tier: MemoryTier,
    ) -> Self {
        Self {
            keys,
            values,
            token_count,
            tier,
        }
    }
}
