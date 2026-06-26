use nerva_core::types::block::address::GlobalBlockAddress;
use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::block::resident::ResidentBlock;
use nerva_core::types::id::allocation::AllocationId;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::memory::MemoryDomainId;

use nerva_core::types::memory::tier::MemoryTier;

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
