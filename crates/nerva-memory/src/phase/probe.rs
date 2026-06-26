use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::block::taxonomy::BlockKind;
use nerva_core::types::error::Result;
use nerva_core::types::id::{DeviceOrdinal, ResidentBlockId, TransportDeviceId};
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::{ExecutionOwner, MutationSemantics};
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::TokenLedger;

use crate::phase::summary::{PhaseHandoffProbeStatus, PhaseHandoffProbeSummary};
use crate::phase::types::{PhaseHandoffPlanner, PhaseHandoffRejectionKind, PhaseHandoffRequest};
use crate::registry::{BlockAllocationRequest, BlockRegistry};

pub fn run_phase_handoff_probe() -> Result<PhaseHandoffProbeSummary> {
    let mut registry = BlockRegistry::new([
        (MemoryTier::Dram, 8 * 1024 * 1024),
        (MemoryTier::Vram, 8 * 1024 * 1024),
        (MemoryTier::PinnedDram, 8 * 1024 * 1024),
    ]);

    let cpu_activation = allocate_phase_block(
        &mut registry,
        BlockKind::Activation,
        MemoryTier::Dram,
        4096,
        ExecutionOwner::Cpu,
        1,
        true,
    )?;
    let gpu_logits = allocate_phase_block(
        &mut registry,
        BlockKind::Logits,
        MemoryTier::Vram,
        4096,
        ExecutionOwner::Gpu(DeviceOrdinal(0)),
        2,
        true,
    )?;
    let gpu_transport = allocate_phase_block(
        &mut registry,
        BlockKind::TransportBuffer,
        MemoryTier::Vram,
        4096,
        ExecutionOwner::Gpu(DeviceOrdinal(0)),
        3,
        true,
    )?;
    let wrong_owner = allocate_phase_block(
        &mut registry,
        BlockKind::Activation,
        MemoryTier::Dram,
        4096,
        ExecutionOwner::Cpu,
        1,
        true,
    )?;
    let stale = allocate_phase_block(
        &mut registry,
        BlockKind::Activation,
        MemoryTier::Dram,
        4096,
        ExecutionOwner::Cpu,
        1,
        true,
    )?;
    let unready = allocate_phase_block(
        &mut registry,
        BlockKind::Activation,
        MemoryTier::Dram,
        4096,
        ExecutionOwner::Cpu,
        1,
        false,
    )?;
    let shared_read_only = allocate_phase_block(
        &mut registry,
        BlockKind::Weight,
        MemoryTier::Dram,
        4096,
        ExecutionOwner::SharedReadOnly,
        1,
        true,
    )?;
    registry
        .block_mut(shared_read_only)
        .expect("probe block exists")
        .semantics = MutationSemantics::Immutable;

    let requests = [
        PhaseHandoffRequest {
            block_id: cpu_activation,
            from: ExecutionOwner::Cpu,
            to: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            required_version: 1,
            reason: "phase_cpu_to_gpu_activation",
        },
        PhaseHandoffRequest {
            block_id: gpu_logits,
            from: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            to: ExecutionOwner::Cpu,
            required_version: 2,
            reason: "phase_gpu_to_cpu_logits",
        },
        PhaseHandoffRequest {
            block_id: gpu_transport,
            from: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            to: ExecutionOwner::Nic(TransportDeviceId(0)),
            required_version: 3,
            reason: "phase_gpu_to_nic_transport",
        },
        PhaseHandoffRequest {
            block_id: wrong_owner,
            from: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            to: ExecutionOwner::Cpu,
            required_version: 1,
            reason: "phase_reject_uncoordinated_writer",
        },
        PhaseHandoffRequest {
            block_id: stale,
            from: ExecutionOwner::Cpu,
            to: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            required_version: 99,
            reason: "phase_reject_stale_version",
        },
        PhaseHandoffRequest {
            block_id: unready,
            from: ExecutionOwner::Cpu,
            to: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            required_version: 1,
            reason: "phase_reject_unready_block",
        },
        PhaseHandoffRequest {
            block_id: shared_read_only,
            from: ExecutionOwner::SharedReadOnly,
            to: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            required_version: 1,
            reason: "phase_reject_shared_read_only_mutation",
        },
    ];

    let plan = PhaseHandoffPlanner::plan(&registry, &requests)?;
    let mut ledger = TokenLedger::new(0);
    let applied = plan.apply(&mut registry, &mut ledger)?;

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

fn allocate_phase_block(
    registry: &mut BlockRegistry,
    kind: BlockKind,
    tier: MemoryTier,
    bytes: usize,
    owner: ExecutionOwner,
    version: u64,
    ready: bool,
) -> Result<ResidentBlockId> {
    let id = registry.allocate(BlockAllocationRequest::new(kind, tier, bytes))?;
    {
        let block = registry.block_mut(id).expect("allocated block exists");
        block.owner = owner;
        block.version = version;
        if !ready {
            block.state = ResidencyState::Prefetching;
        }
    }
    if ready {
        registry.mark_ready(id)?;
    }
    Ok(id)
}
