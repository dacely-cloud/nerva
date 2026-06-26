use nerva_core::types::block::address::GlobalBlockAddress;
use nerva_core::types::memory::tier::MemoryTier;

use crate::arena::kind::ArenaKind;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArenaReservation {
    pub offset: usize,
    pub bytes: usize,
    pub align: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArenaCheckpoint {
    pub(crate) used: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArenaRegion {
    pub name: &'static str,
    pub kind: ArenaKind,
    pub tier: MemoryTier,
    pub address: GlobalBlockAddress,
    pub offset: usize,
    pub bytes: usize,
    pub align: usize,
}
