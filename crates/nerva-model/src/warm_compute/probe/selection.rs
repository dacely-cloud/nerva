use nerva_core::types::error::{NervaError, Result};

use crate::warm_compute::strategy::WarmComputeStrategy;
use crate::warm_compute::summary::WarmComputeCandidate;

pub(crate) fn candidate_visible_ns(
    candidates: &[WarmComputeCandidate],
    strategy: WarmComputeStrategy,
) -> Result<u64> {
    candidates
        .iter()
        .find(|candidate| candidate.strategy == strategy)
        .map(|candidate| candidate.visible_ns)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("missing warm compute candidate {}", strategy.label()),
        })
}
