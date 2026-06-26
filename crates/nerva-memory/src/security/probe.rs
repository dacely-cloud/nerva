use nerva_core::types::block::flags::BlockFlags;
use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::Result;
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::transport::TransportDeviceId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::registry::request::BlockAllocationRequest;
use crate::registry::table::registry::BlockRegistry;
use crate::security::policy::SanitizationPhase;
use crate::security::sanitizer::{sanitize_sensitive_block, stale_sensitive_version_revoked};
use crate::security::summary::{SecurityIsolationStatus, SecurityIsolationSummary};

pub fn run_security_isolation_probe() -> Result<SecurityIsolationSummary> {
    let mut registry = BlockRegistry::new([
        (MemoryTier::Dram, 8 * 1024 * 1024),
        (MemoryTier::Vram, 8 * 1024 * 1024),
        (MemoryTier::PinnedDram, 8 * 1024 * 1024),
    ]);
    let token_state = allocate_probe_block(
        &mut registry,
        BlockKind::TokenState,
        MemoryTier::PinnedDram,
        4096,
        ExecutionOwner::Cpu,
        7,
        true,
        true,
    )?;
    let activation = allocate_probe_block(
        &mut registry,
        BlockKind::Activation,
        MemoryTier::Vram,
        8192,
        ExecutionOwner::Gpu(DeviceOrdinal(0)),
        11,
        true,
        true,
    )?;
    let transport = allocate_probe_block(
        &mut registry,
        BlockKind::TransportBuffer,
        MemoryTier::PinnedDram,
        4096,
        ExecutionOwner::Nic(TransportDeviceId(0)),
        5,
        true,
        true,
    )?;
    let non_sensitive = allocate_probe_block(
        &mut registry,
        BlockKind::Metadata,
        MemoryTier::Dram,
        1024,
        ExecutionOwner::Cpu,
        1,
        true,
        false,
    )?;
    let unready_sensitive = allocate_probe_block(
        &mut registry,
        BlockKind::Workspace,
        MemoryTier::Dram,
        1024,
        ExecutionOwner::Cpu,
        1,
        false,
        true,
    )?;

    let mut ledger = TokenLedger::new(0);
    let hot_path_version_before = registry
        .block(token_state)
        .expect("probe block exists")
        .version;
    let hot_path_sanitize_rejections = u64::from(
        sanitize_sensitive_block(
            &mut registry,
            token_state,
            SanitizationPhase::HotPath,
            &mut ledger,
        )
        .is_err()
            && registry
                .block(token_state)
                .expect("probe block exists")
                .version
                == hot_path_version_before,
    );
    let non_sensitive_rejections = u64::from(
        sanitize_sensitive_block(
            &mut registry,
            non_sensitive,
            SanitizationPhase::Maintenance,
            &mut ledger,
        )
        .is_err(),
    );
    let unready_rejections = u64::from(
        sanitize_sensitive_block(
            &mut registry,
            unready_sensitive,
            SanitizationPhase::Maintenance,
            &mut ledger,
        )
        .is_err(),
    );

    let mut outcomes = Vec::new();
    for block_id in [token_state, activation, transport] {
        outcomes.push(sanitize_sensitive_block(
            &mut registry,
            block_id,
            SanitizationPhase::Maintenance,
            &mut ledger,
        )?);
    }

    let mut stale_version_rejections = 0u64;
    let mut ready_after_sanitize = 0u64;
    let mut owner_cleared_after_sanitize = 0u64;
    let mut bytes_sanitized = 0usize;
    for outcome in &outcomes {
        bytes_sanitized = bytes_sanitized.saturating_add(outcome.bytes);
        if stale_sensitive_version_revoked(&registry, outcome.block_id, outcome.old_version)? {
            stale_version_rejections = stale_version_rejections.saturating_add(1);
        }
        let block = registry
            .block(outcome.block_id)
            .expect("sanitized probe block exists");
        if block.state == ResidencyState::Ready {
            ready_after_sanitize = ready_after_sanitize.saturating_add(1);
        }
        if block.owner == ExecutionOwner::None {
            owner_cleared_after_sanitize = owner_cleared_after_sanitize.saturating_add(1);
        }
    }

    ledger.require_classified_syncs()?;
    ledger.require_zero_hot_path_allocations()?;

    Ok(SecurityIsolationSummary {
        status: SecurityIsolationStatus::Ok,
        sensitive_blocks: outcomes.len() as u64,
        bytes_sanitized,
        zero_fill_events: ledger.event_count(LedgerEventKind::CpuActivity),
        version_revocations: ledger.sync_count_for(SyncClass::PhaseHandoff),
        hot_path_sanitize_rejections,
        non_sensitive_rejections,
        unready_rejections,
        stale_version_rejections,
        ready_after_sanitize,
        owner_cleared_after_sanitize,
        hot_path_allocations: ledger.hot_path_allocations,
        error: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn allocate_probe_block(
    registry: &mut BlockRegistry,
    kind: BlockKind,
    tier: MemoryTier,
    bytes: usize,
    owner: ExecutionOwner,
    version: u64,
    ready: bool,
    sensitive: bool,
) -> Result<ResidentBlockId> {
    let id = registry.allocate(BlockAllocationRequest::new(kind, tier, bytes))?;
    {
        let block = registry.block_mut(id).expect("allocated block exists");
        block.owner = owner;
        block.version = version;
        if sensitive {
            block.flags = BlockFlags::from_bits(block.flags.bits() | BlockFlags::SENSITIVE);
        }
        if !ready {
            block.state = ResidencyState::Prefetching;
        }
    }
    if ready {
        registry.mark_ready(id)?;
    }
    Ok(id)
}
