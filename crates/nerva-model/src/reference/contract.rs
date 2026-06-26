use nerva_core::types::block::BlockKind;
use nerva_core::types::dtype::DType;
use nerva_core::types::memory::MemoryTier;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ModelBlockContract {
    pub block_kind: BlockKind,
    pub weight_dtype: DType,
    pub activation_dtype: DType,
    pub weight_tier: MemoryTier,
    pub activation_tier: MemoryTier,
}

impl ModelBlockContract {
    pub const fn reference_f32() -> Self {
        Self {
            block_kind: BlockKind::Weight,
            weight_dtype: DType::F32,
            activation_dtype: DType::F32,
            weight_tier: MemoryTier::Dram,
            activation_tier: MemoryTier::Dram,
        }
    }
}
