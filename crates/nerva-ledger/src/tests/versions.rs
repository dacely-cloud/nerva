use nerva_core::types::id::block::ResidentBlockId;

use crate::types::decision::BlockVersionDependency;
use crate::types::token::ledger::TokenLedger;

#[test]
fn block_version_dependencies_validate_observed_versions() {
    let mut ledger = TokenLedger::new(0);
    ledger.record_block_version_dependency(BlockVersionDependency {
        block_id: ResidentBlockId(7),
        required_version: 2,
        observed_version: 2,
        label: "weight_step",
    });
    ledger.record_block_version_dependency(BlockVersionDependency {
        block_id: ResidentBlockId(8),
        required_version: 2,
        observed_version: 3,
        label: "newer_replica",
    });

    assert_eq!(ledger.block_version_dependencies.len(), 2);
    assert!(ledger.require_satisfied_block_versions().is_ok());
}

#[test]
fn block_version_dependencies_reject_stale_observations() {
    let mut ledger = TokenLedger::new(0);
    ledger.record_block_version_dependency(BlockVersionDependency {
        block_id: ResidentBlockId(7),
        required_version: 4,
        observed_version: 3,
        label: "stale_weight_step",
    });

    assert!(ledger.require_satisfied_block_versions().is_err());
}
