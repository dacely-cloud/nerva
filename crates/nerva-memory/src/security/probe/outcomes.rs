use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::Result;
use nerva_core::types::ownership::owner::ExecutionOwner;

use crate::registry::table::registry::BlockRegistry;
use crate::security::sanitizer::{SensitiveSanitizationOutcome, stale_sensitive_version_revoked};

pub(super) struct SanitizedOutcomeCounts {
    pub bytes_sanitized: usize,
    pub stale_version_rejections: u64,
    pub ready_after_sanitize: u64,
    pub owner_cleared_after_sanitize: u64,
}

pub(super) fn count_sanitized_outcomes(
    registry: &BlockRegistry,
    outcomes: &[SensitiveSanitizationOutcome],
) -> Result<SanitizedOutcomeCounts> {
    let mut counts = SanitizedOutcomeCounts {
        bytes_sanitized: 0,
        stale_version_rejections: 0,
        ready_after_sanitize: 0,
        owner_cleared_after_sanitize: 0,
    };

    for outcome in outcomes {
        counts.bytes_sanitized = counts.bytes_sanitized.saturating_add(outcome.bytes);
        if stale_sensitive_version_revoked(registry, outcome.block_id, outcome.old_version)? {
            counts.stale_version_rejections = counts.stale_version_rejections.saturating_add(1);
        }
        let block = registry
            .block(outcome.block_id)
            .expect("sanitized probe block exists");
        if block.state == ResidencyState::Ready {
            counts.ready_after_sanitize = counts.ready_after_sanitize.saturating_add(1);
        }
        if block.owner == ExecutionOwner::None {
            counts.owner_cleared_after_sanitize =
                counts.owner_cleared_after_sanitize.saturating_add(1);
        }
    }

    Ok(counts)
}
