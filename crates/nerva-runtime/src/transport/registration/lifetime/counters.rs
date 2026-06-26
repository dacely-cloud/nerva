use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::transport::registration::lifetime::summary::{
    TransportRegistrationLifecycleStatus, TransportRegistrationLifecycleSummary,
};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct LifecycleCounters {
    pub explicit_key_revocations: u64,
    pub block_lifetime_revocations: u64,
    pub destroy_revocations: u64,
    pub lookup_hits_before_revoke: u64,
    pub post_revoke_misses: u64,
    pub safe_move_post_revoke_misses: u64,
    pub stale_mapping_reuse_rejections: u64,
}

impl LifecycleCounters {
    pub(super) fn total_revocations(self) -> u64 {
        self.explicit_key_revocations
            .saturating_add(self.block_lifetime_revocations)
            .saturating_add(self.destroy_revocations)
    }

    pub(super) fn stale_handle_reuse_prevented(self) -> bool {
        self.post_revoke_misses > 0
            && self.safe_move_post_revoke_misses > 0
            && self.stale_mapping_reuse_rejections > 0
    }

    pub(super) fn summary(
        self,
        bootstrap_registrations: u64,
        registered_before_revoke: u64,
        final_registered_entries: u64,
        ledger: &TokenLedger,
    ) -> TransportRegistrationLifecycleSummary {
        let total_revocations = self.total_revocations();
        TransportRegistrationLifecycleSummary {
            status: TransportRegistrationLifecycleStatus::Ok,
            bootstrap_registrations,
            registered_before_revoke,
            explicit_key_revocations: self.explicit_key_revocations,
            block_lifetime_revocations: self.block_lifetime_revocations,
            destroy_revocations: self.destroy_revocations,
            total_revocations,
            final_registered_entries,
            lookup_hits_before_revoke: self.lookup_hits_before_revoke,
            post_revoke_misses: self.post_revoke_misses,
            safe_move_post_revoke_misses: self.safe_move_post_revoke_misses,
            stale_mapping_reuse_rejections: self.stale_mapping_reuse_rejections,
            revocation_syncs: total_revocations,
            phase_handoff_syncs: ledger.sync_count_for(SyncClass::PhaseHandoff),
            transport_events: ledger.event_count(LedgerEventKind::Transport),
            stale_handle_reuse_prevented: self.stale_handle_reuse_prevented(),
            hot_path_allocations: ledger.hot_path_allocations,
            error: None,
        }
    }
}
