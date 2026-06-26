use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::block::ResidentBlockId;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::arena::kind::{AllocationPhase, ArenaKind};
use crate::arena::set::static_set::StaticArenaSet;
use crate::registry::request::BlockAllocationRequest;
use crate::registry::table::registry::BlockRegistry;

impl StaticArenaSet {
    pub fn reject_hot_path_reservation_with_ledger(
        &mut self,
        kind: ArenaKind,
        name: &'static str,
        bytes: usize,
        align: usize,
        ledger: &mut TokenLedger,
    ) -> Result<()> {
        let before = self.arena_mut(kind).used();
        match self.reserve(kind, name, bytes, align, AllocationPhase::HotPath) {
            Ok(_) => Err(NervaError::InvalidArgument {
                reason: "static arena accepted forbidden hot-path reservation".to_string(),
            }),
            Err(err) => {
                debug_assert_eq!(self.arena_mut(kind).used(), before);
                ledger.record_hot_path_allocation_attempt(name, bytes, kind.tier());
                Err(err)
            }
        }
    }

    pub fn reserve_resident_block(
        &mut self,
        registry: &mut BlockRegistry,
        kind: ArenaKind,
        name: &'static str,
        request: BlockAllocationRequest,
        align: usize,
        phase: AllocationPhase,
    ) -> Result<ResidentBlockId> {
        if request.tier != kind.tier() {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "arena kind {:?} cannot reserve block requested for {:?}",
                    kind, request.tier
                ),
            });
        }

        let checkpoint = self.arena_mut(kind).checkpoint();
        let region = self.reserve(kind, name, request.bytes, align, phase)?;
        match registry.allocate(request) {
            Ok(id) => {
                registry.bind_address(id, region.address)?;
                Ok(id)
            }
            Err(err) => {
                let _ = self.arena_mut(kind).restore(checkpoint);
                Err(err)
            }
        }
    }
}
