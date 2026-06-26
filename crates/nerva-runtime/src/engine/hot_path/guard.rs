use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_memory::arena::kind::ArenaKind;
use nerva_memory::arena::set::StaticArenaSet;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HotPathGuard {
    ledger: TokenLedger,
    entered_scopes: u64,
    exited_scopes: u64,
    active_scopes: u64,
    forbidden_allocation_attempts: u64,
    rejected_allocation_attempts: u64,
    attempted_bytes: usize,
    release_to_system_calls: u64,
    usage_preserved_after_rejections: bool,
}

impl HotPathGuard {
    pub fn new(token_index: u64) -> Self {
        Self {
            ledger: TokenLedger::new(token_index),
            entered_scopes: 0,
            exited_scopes: 0,
            active_scopes: 0,
            forbidden_allocation_attempts: 0,
            rejected_allocation_attempts: 0,
            attempted_bytes: 0,
            release_to_system_calls: 0,
            usage_preserved_after_rejections: true,
        }
    }

    pub fn enter(&mut self, label: &'static str) -> Result<HotPathScope<'_>> {
        if self.active_scopes != 0 {
            return Err(NervaError::InvalidArgument {
                reason: "cannot enter nested hot-path guard scope".to_string(),
            });
        }
        self.entered_scopes += 1;
        self.active_scopes = 1;
        Ok(HotPathScope { guard: self, label })
    }

    pub const fn entered_scopes(&self) -> u64 {
        self.entered_scopes
    }

    pub const fn exited_scopes(&self) -> u64 {
        self.exited_scopes
    }

    pub const fn active_scopes(&self) -> u64 {
        self.active_scopes
    }

    pub const fn forbidden_allocation_attempts(&self) -> u64 {
        self.forbidden_allocation_attempts
    }

    pub const fn rejected_allocation_attempts(&self) -> u64 {
        self.rejected_allocation_attempts
    }

    pub const fn attempted_bytes(&self) -> usize {
        self.attempted_bytes
    }

    pub const fn release_to_system_calls(&self) -> u64 {
        self.release_to_system_calls
    }

    pub const fn usage_preserved_after_rejections(&self) -> bool {
        self.usage_preserved_after_rejections
    }

    pub fn ledger(&self) -> &TokenLedger {
        &self.ledger
    }

    fn leave(&mut self) {
        self.active_scopes = 0;
        self.exited_scopes += 1;
    }
}

pub struct HotPathScope<'a> {
    guard: &'a mut HotPathGuard,
    label: &'static str,
}

impl HotPathScope<'_> {
    pub const fn label(&self) -> &'static str {
        self.label
    }

    pub fn reject_nested_scope(&self) -> Result<()> {
        Err(NervaError::InvalidArgument {
            reason: format!("hot-path scope '{}' is already active", self.label),
        })
    }

    pub fn reject_arena_reservation(
        &mut self,
        arenas: &mut StaticArenaSet,
        kind: ArenaKind,
        name: &'static str,
        bytes: usize,
        align: usize,
    ) -> Result<()> {
        let before = arena_used(arenas, kind);
        self.guard.forbidden_allocation_attempts += 1;
        self.guard.attempted_bytes = self.guard.attempted_bytes.saturating_add(bytes);
        match arenas.reject_hot_path_reservation_with_ledger(
            kind,
            name,
            bytes,
            align,
            &mut self.guard.ledger,
        ) {
            Ok(()) => Err(NervaError::InvalidArgument {
                reason: "hot-path guard accepted a forbidden arena reservation".to_string(),
            }),
            Err(err) => {
                let after = arena_used(arenas, kind);
                if after == before {
                    self.guard.rejected_allocation_attempts += 1;
                } else {
                    self.guard.usage_preserved_after_rejections = false;
                }
                Err(err)
            }
        }
    }
}

impl Drop for HotPathScope<'_> {
    fn drop(&mut self) {
        self.guard.leave();
    }
}

pub fn allocation_event_count(guard: &HotPathGuard) -> u64 {
    guard.ledger().event_count(LedgerEventKind::Allocation)
}

fn arena_used(arenas: &StaticArenaSet, kind: ArenaKind) -> usize {
    match kind {
        ArenaKind::Device => arenas.device().used(),
        ArenaKind::PinnedHost => arenas.pinned_host().used(),
        ArenaKind::Host => arenas.host().used(),
    }
}
