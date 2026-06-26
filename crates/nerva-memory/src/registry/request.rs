use nerva_core::types::block::taxonomy::BlockKind;
use nerva_core::types::dtype::DType;
use nerva_core::types::id::LayoutId;
use nerva_core::types::memory::MemoryTier;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockAllocationRequest {
    pub kind: BlockKind,
    pub tier: MemoryTier,
    pub bytes: usize,
    pub dtype: DType,
    pub layout: LayoutId,
}

impl BlockAllocationRequest {
    pub const fn new(kind: BlockKind, tier: MemoryTier, bytes: usize) -> Self {
        Self {
            kind,
            tier,
            bytes,
            dtype: DType::U8,
            layout: LayoutId(0),
        }
    }

    pub const fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub const fn with_layout(mut self, layout: LayoutId) -> Self {
        self.layout = layout;
        self
    }
}
