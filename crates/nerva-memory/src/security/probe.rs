use nerva_core::types::error::Result;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::security::policy::SanitizationPhase;
use crate::security::probe::fixture::SecurityProbeFixture;
use crate::security::probe::outcomes::count_sanitized_outcomes;
use crate::security::sanitizer::sanitize_sensitive_block;
use crate::security::summary::{SecurityIsolationStatus, SecurityIsolationSummary};

mod fixture;
mod outcomes;

pub fn run_security_isolation_probe() -> Result<SecurityIsolationSummary> {
    let mut fixture = SecurityProbeFixture::allocate()?;
    let mut ledger = TokenLedger::new(0);
    let hot_path_version_before = fixture
        .registry
        .block(fixture.token_state)
        .expect("probe block exists")
        .version;
    let hot_path_sanitize_rejections = u64::from(
        sanitize_sensitive_block(
            &mut fixture.registry,
            fixture.token_state,
            SanitizationPhase::HotPath,
            &mut ledger,
        )
        .is_err()
            && fixture
                .registry
                .block(fixture.token_state)
                .expect("probe block exists")
                .version
                == hot_path_version_before,
    );
    let non_sensitive_rejections = u64::from(
        sanitize_sensitive_block(
            &mut fixture.registry,
            fixture.non_sensitive,
            SanitizationPhase::Maintenance,
            &mut ledger,
        )
        .is_err(),
    );
    let unready_rejections = u64::from(
        sanitize_sensitive_block(
            &mut fixture.registry,
            fixture.unready_sensitive,
            SanitizationPhase::Maintenance,
            &mut ledger,
        )
        .is_err(),
    );

    let mut outcomes = Vec::new();
    for block_id in fixture.sensitive_blocks() {
        outcomes.push(sanitize_sensitive_block(
            &mut fixture.registry,
            block_id,
            SanitizationPhase::Maintenance,
            &mut ledger,
        )?);
    }
    let counts = count_sanitized_outcomes(&fixture.registry, &outcomes)?;

    ledger.require_classified_syncs()?;
    ledger.require_zero_hot_path_allocations()?;

    Ok(SecurityIsolationSummary {
        status: SecurityIsolationStatus::Ok,
        sensitive_blocks: outcomes.len() as u64,
        bytes_sanitized: counts.bytes_sanitized,
        zero_fill_events: ledger.event_count(LedgerEventKind::CpuActivity),
        version_revocations: ledger.sync_count_for(SyncClass::PhaseHandoff),
        hot_path_sanitize_rejections,
        non_sensitive_rejections,
        unready_rejections,
        stale_version_rejections: counts.stale_version_rejections,
        ready_after_sanitize: counts.ready_after_sanitize,
        owner_cleared_after_sanitize: counts.owner_cleared_after_sanitize,
        hot_path_allocations: ledger.hot_path_allocations,
        error: None,
    })
}
