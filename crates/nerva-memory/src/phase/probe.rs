use nerva_core::types::error::Result;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::phase::probe::fixture::PhaseProbeFixture;
use crate::phase::probe::requests::phase_handoff_requests;
use crate::phase::summary::{PhaseHandoffProbeStatus, PhaseHandoffProbeSummary};
use crate::phase::types::{PhaseHandoffPlanner, PhaseHandoffRejectionKind};

mod fixture;
mod requests;

pub fn run_phase_handoff_probe() -> Result<PhaseHandoffProbeSummary> {
    let mut fixture = PhaseProbeFixture::allocate()?;
    let requests = phase_handoff_requests(&fixture);

    let plan = PhaseHandoffPlanner::plan(&fixture.registry, &requests)?;
    let mut ledger = TokenLedger::new(0);
    let applied = plan.apply(&mut fixture.registry, &mut ledger)?;

    Ok(PhaseHandoffProbeSummary {
        status: PhaseHandoffProbeStatus::Ok,
        planned_handoffs: plan.entries.len() as u64,
        applied_handoffs: applied.applied_handoffs,
        rejected_handoffs: plan.rejections.len() as u64,
        owner_mismatch_rejections: plan.rejected_count(PhaseHandoffRejectionKind::OwnerMismatch),
        stale_version_rejections: plan.rejected_count(PhaseHandoffRejectionKind::StaleVersion),
        unready_rejections: plan.rejected_count(PhaseHandoffRejectionKind::BlockNotReady),
        illegal_transition_rejections: plan
            .rejected_count(PhaseHandoffRejectionKind::IllegalTransition),
        phase_handoff_syncs: ledger.sync_count_for(SyncClass::PhaseHandoff),
        version_publications: applied.version_publications,
        final_max_version: applied.final_max_version,
        hot_path_allocations: ledger.hot_path_allocations,
        error: None,
    })
}
