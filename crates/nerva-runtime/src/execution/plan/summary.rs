use nerva_core::types::error::Result;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::execution::summary::{ExecutionTransactionStatus, ExecutionTransactionSummary};
use crate::execution::types::ExecutionTransactionSpec;

pub fn summarize_transaction(
    spec: &ExecutionTransactionSpec,
    ledger: &TokenLedger,
) -> Result<ExecutionTransactionSummary> {
    let operations = spec.operations.len() as u64;
    let graph_capturable_operations = spec
        .operations
        .iter()
        .filter(|operation| operation.graph_capturable)
        .count() as u64;
    let cpu_operations = spec
        .operations
        .iter()
        .filter(|operation| operation.executor == ExecutionOwner::Cpu)
        .count() as u64;
    let gpu_operations = spec
        .operations
        .iter()
        .filter(|operation| matches!(operation.executor, ExecutionOwner::Gpu(_)))
        .count() as u64;
    let block_uses = spec
        .operations
        .iter()
        .map(|operation| operation.block_uses.len() as u64)
        .sum::<u64>();
    let device = spec
        .operations
        .iter()
        .find_map(|operation| match operation.executor {
            ExecutionOwner::Gpu(device) => Some(device),
            _ => None,
        });
    let (device_active_ns, gpu_idle_ns) = match device {
        Some(device) => (
            ledger.device_active_ns(device)?,
            ledger.device_idle_ns(device)?,
        ),
        None => (0, 0),
    };
    let stale_dependencies = ledger
        .block_version_dependencies
        .iter()
        .filter(|dependency| dependency.observed_version < dependency.required_version)
        .count() as u64;
    let unclassified_syncs = ledger
        .events
        .iter()
        .filter(|event| event.kind == LedgerEventKind::Sync && event.sync_class.is_none())
        .count() as u64;

    Ok(ExecutionTransactionSummary {
        status: ExecutionTransactionStatus::Ok,
        operations,
        graph_capturable_operations,
        cpu_operations,
        gpu_operations,
        block_uses,
        block_version_dependencies: ledger.block_version_dependencies.len() as u64,
        execution_decisions: ledger.execution_decisions.len() as u64,
        hard_syncs: ledger.sync_count_for(SyncClass::HardSync),
        soft_visibility_syncs: ledger.sync_count_for(SyncClass::SoftVisibilitySync),
        phase_handoff_syncs: ledger.sync_count_for(SyncClass::PhaseHandoff),
        policy_syncs: ledger.sync_count_for(SyncClass::PolicySync),
        debug_syncs: ledger.sync_count_for(SyncClass::DebugSync),
        graph_replay_events: ledger.event_count(LedgerEventKind::GraphReplay),
        kernel_launch_events: ledger.event_count(LedgerEventKind::KernelLaunch),
        device_activity_events: ledger.event_count(LedgerEventKind::DeviceActivity),
        cpu_activity_events: ledger.event_count(LedgerEventKind::CpuActivity),
        device_active_ns,
        gpu_idle_ns,
        host_event_wait_ns: ledger.sync_latency_ns_for(SyncClass::SoftVisibilitySync),
        total_predicted_visible_ns: spec
            .operations
            .iter()
            .map(|operation| operation.predicted_visible_ns)
            .sum(),
        hot_path_allocations: ledger.hot_path_allocations,
        stale_dependencies,
        unclassified_syncs,
        error: None,
    })
}
