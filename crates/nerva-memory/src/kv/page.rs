use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;

use crate::arena::kind::ArenaKind;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvPageSpec {
    pub layer_id: u32,
    pub head_group_id: u32,
    pub block_size_tokens: u32,
    pub page_bytes: usize,
    pub tier: MemoryTier,
    pub arena_kind: ArenaKind,
    pub align: usize,
}

impl KvPageSpec {
    pub const fn new(
        layer_id: u32,
        head_group_id: u32,
        block_size_tokens: u32,
        page_bytes: usize,
        tier: MemoryTier,
        arena_kind: ArenaKind,
        align: usize,
    ) -> Self {
        Self {
            layer_id,
            head_group_id,
            block_size_tokens,
            page_bytes,
            tier,
            arena_kind,
            align,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct KvPrefixKey {
    pub hash: [u8; 32],
    pub group_id: u32,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvPageHandle {
    pub page_index: u32,
    pub block_id: ResidentBlockId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvPageDescriptor {
    pub page_index: u32,
    pub block_id: ResidentBlockId,
    pub layer_id: u32,
    pub head_group_id: u32,
    pub token_start: u32,
    pub token_count: u32,
    pub block_size_tokens: u32,
    pub page_bytes: usize,
    pub ref_count: u32,
    pub prefix_key: Option<KvPrefixKey>,
    pub prefix_tokens: Option<u32>,
    pub last_use: u64,
    pub next_use: Option<u64>,
    pub(crate) is_free: bool,
}
