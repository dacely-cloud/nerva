use nerva_core::types::block::flags::BlockFlags;
use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::registry::table::registry::BlockRegistry;
use crate::security::policy::SanitizationPhase;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SensitiveSanitizationOutcome {
    pub block_id: ResidentBlockId,
    pub old_version: u64,
    pub new_version: u64,
    pub bytes: usize,
    pub tier: MemoryTier,
}

pub fn sanitize_sensitive_block(
    registry: &mut BlockRegistry,
    block_id: ResidentBlockId,
    phase: SanitizationPhase,
    ledger: &mut TokenLedger,
) -> Result<SensitiveSanitizationOutcome> {
    if phase == SanitizationPhase::HotPath {
        return Err(NervaError::InvalidArgument {
            reason: "sensitive block sanitization is forbidden on the hot path".to_string(),
        });
    }

    let block = registry
        .block_mut(block_id)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("unknown sensitive block id {}", block_id.0),
        })?;
    if !block.flags.contains(BlockFlags::SENSITIVE) {
        return Err(NervaError::InvalidArgument {
            reason: "block is not marked sensitive".to_string(),
        });
    }
    if block.state != ResidencyState::Ready {
        return Err(NervaError::ResidencyViolation {
            block_id,
            reason: "sensitive block must be ready before sanitization".to_string(),
        });
    }

    let old_version = block.version;
    let bytes = block.bytes;
    let tier = block.tier;
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::CpuActivity,
        sync_class: None,
        metric_source: MetricSource::RuntimeTimestamp,
        block_id: Some(block_id),
        from_tier: Some(tier),
        to_tier: Some(tier),
        bytes,
        latency_ns: 1,
        label: "security_sensitive_zero_fill",
    });
    ledger.record_sync(
        SyncClass::PhaseHandoff,
        Some(block_id),
        Some(tier),
        Some(tier),
        bytes,
        1,
        MetricSource::RuntimeTimestamp,
        "security_sensitive_version_revoke",
    );
    let new_version = block.publish(ExecutionOwner::None);

    Ok(SensitiveSanitizationOutcome {
        block_id,
        old_version,
        new_version,
        bytes,
        tier,
    })
}

pub fn stale_sensitive_version_revoked(
    registry: &BlockRegistry,
    block_id: ResidentBlockId,
    stale_version: u64,
) -> Result<bool> {
    let block = registry
        .block(block_id)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("unknown sensitive block id {}", block_id.0),
        })?;
    Ok(block.version != stale_version)
}
