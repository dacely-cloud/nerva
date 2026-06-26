use nerva_core::types::block::{BlockKind, GlobalBlockAddress, ResidentBlock};
use nerva_core::types::id::{AllocationId, MemoryDomainId, ResidentBlockId};
use nerva_core::types::memory::MemoryTier;

use crate::arena::region::ArenaReservation;

pub fn resident_block_for_reservation(
    id: ResidentBlockId,
    kind: BlockKind,
    reservation: ArenaReservation,
) -> ResidentBlock {
    ResidentBlock::new(id, kind, MemoryTier::Dram, reservation.bytes).with_address(
        GlobalBlockAddress {
            domain: MemoryDomainId::CPU_DRAM,
            allocation: AllocationId(id.0),
            offset: reservation.offset as u64,
        },
    )
}
