use nerva_core::types::block::taxonomy::BlockKind;
use nerva_core::types::id::{DeviceOrdinal, TransportDeviceId};
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::TokenLedger;

use crate::phase::probe::run_phase_handoff_probe;
use crate::phase::types::{PhaseHandoffPlanner, PhaseHandoffRejectionKind, PhaseHandoffRequest};
use crate::registry::{BlockAllocationRequest, BlockRegistry};

#[test]
fn phase_handoff_probe_records_handoffs_and_rejections() {
    let summary = run_phase_handoff_probe().unwrap();

    assert!(summary.passed());
    assert_eq!(summary.planned_handoffs, 3);
    assert_eq!(summary.applied_handoffs, 3);
    assert_eq!(summary.rejected_handoffs, 4);
    assert_eq!(summary.owner_mismatch_rejections, 1);
    assert_eq!(summary.stale_version_rejections, 1);
    assert_eq!(summary.unready_rejections, 1);
    assert_eq!(summary.illegal_transition_rejections, 1);
    assert_eq!(summary.phase_handoff_syncs, 3);
    assert_eq!(summary.version_publications, 3);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"phase_handoff_syncs\":3"));
    assert!(
        summary
            .to_json()
            .contains("\"owner_mismatch_rejections\":1")
    );
}

#[test]
fn phase_handoff_plan_rejects_stale_plan_on_apply() {
    let mut registry = BlockRegistry::new([(MemoryTier::Dram, 1024 * 1024)]);
    let block_id = registry
        .allocate(BlockAllocationRequest::new(
            BlockKind::Activation,
            MemoryTier::Dram,
            1024,
        ))
        .unwrap();
    {
        let block = registry.block_mut(block_id).unwrap();
        block.owner = ExecutionOwner::Cpu;
        block.version = 1;
    }
    registry.mark_ready(block_id).unwrap();

    let plan = PhaseHandoffPlanner::plan(
        &registry,
        &[PhaseHandoffRequest {
            block_id,
            from: ExecutionOwner::Cpu,
            to: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            required_version: 1,
            reason: "test_phase_handoff",
        }],
    )
    .unwrap();
    assert_eq!(plan.entries.len(), 1);

    registry
        .block_mut(block_id)
        .unwrap()
        .publish(ExecutionOwner::Cpu);
    let err = plan
        .apply(&mut registry, &mut TokenLedger::new(0))
        .unwrap_err();
    assert!(format!("{err:?}").contains("stale"));
}

#[test]
fn phase_handoff_planner_rejects_uncoordinated_writer() {
    let mut registry = BlockRegistry::new([(MemoryTier::Vram, 1024 * 1024)]);
    let block_id = registry
        .allocate(BlockAllocationRequest::new(
            BlockKind::TransportBuffer,
            MemoryTier::Vram,
            1024,
        ))
        .unwrap();
    {
        let block = registry.block_mut(block_id).unwrap();
        block.owner = ExecutionOwner::Gpu(DeviceOrdinal(0));
        block.version = 7;
    }
    registry.mark_ready(block_id).unwrap();

    let plan = PhaseHandoffPlanner::plan(
        &registry,
        &[PhaseHandoffRequest {
            block_id,
            from: ExecutionOwner::Cpu,
            to: ExecutionOwner::Nic(TransportDeviceId(0)),
            required_version: 7,
            reason: "test_reject_wrong_owner",
        }],
    )
    .unwrap();
    assert_eq!(plan.entries.len(), 0);
    assert_eq!(
        plan.rejected_count(PhaseHandoffRejectionKind::OwnerMismatch),
        1
    );
}

#[test]
fn phase_handoff_apply_publishes_new_owner_and_sync() {
    let mut registry = BlockRegistry::new([(MemoryTier::Vram, 1024 * 1024)]);
    let block_id = registry
        .allocate(BlockAllocationRequest::new(
            BlockKind::TransportBuffer,
            MemoryTier::Vram,
            1024,
        ))
        .unwrap();
    {
        let block = registry.block_mut(block_id).unwrap();
        block.owner = ExecutionOwner::Gpu(DeviceOrdinal(0));
        block.version = 3;
    }
    registry.mark_ready(block_id).unwrap();

    let plan = PhaseHandoffPlanner::plan(
        &registry,
        &[PhaseHandoffRequest {
            block_id,
            from: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            to: ExecutionOwner::Nic(TransportDeviceId(0)),
            required_version: 3,
            reason: "test_gpu_to_nic",
        }],
    )
    .unwrap();
    let mut ledger = TokenLedger::new(0);
    let applied = plan.apply(&mut registry, &mut ledger).unwrap();

    assert_eq!(applied.applied_handoffs, 1);
    assert_eq!(applied.version_publications, 1);
    assert_eq!(ledger.sync_count_for(SyncClass::PhaseHandoff), 1);
    let block = registry.block(block_id).unwrap();
    assert_eq!(block.owner, ExecutionOwner::Nic(TransportDeviceId(0)));
    assert_eq!(block.version, 4);
}
