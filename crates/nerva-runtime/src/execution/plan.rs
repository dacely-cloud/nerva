use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::decision::{BlockVersionDependency, CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{DeviceTimelineSpan, LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::TokenLedger;
use nerva_memory::registry::BlockRegistry;

use crate::execution::summary::{ExecutionTransactionStatus, ExecutionTransactionSummary};
use crate::execution::types::{
    ExecutionTransactionSpec, TransactionOperation, TransactionOperationKind,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionTransactionPlan {
    pub spec: ExecutionTransactionSpec,
    pub ledger: TokenLedger,
    pub summary: ExecutionTransactionSummary,
}

pub fn plan_execution_transaction(
    spec: ExecutionTransactionSpec,
    registry: &BlockRegistry,
) -> Result<ExecutionTransactionPlan> {
    if spec.operations.is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: "execution transaction requires at least one operation".to_string(),
        });
    }

    let mut ledger = TokenLedger::new(spec.token_index);
    let mut clock_ns = 0u64;

    for operation in &spec.operations {
        validate_operation(operation)?;
        record_execution_decision(&mut ledger, operation);
        record_operation_events(&mut ledger, operation, clock_ns)?;
        validate_and_record_block_uses(&mut ledger, registry, operation)?;
        clock_ns = clock_ns.saturating_add(operation.predicted_visible_ns);
    }

    ledger.require_satisfied_block_versions()?;
    ledger.require_classified_syncs()?;
    ledger.require_zero_hot_path_allocations()?;

    let summary = summarize_transaction(&spec, &ledger)?;
    Ok(ExecutionTransactionPlan {
        spec,
        ledger,
        summary,
    })
}

fn validate_operation(operation: &TransactionOperation) -> Result<()> {
    if operation.name.is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: "transaction operation name must be non-empty".to_string(),
        });
    }
    if operation.predicted_visible_ns == 0 {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "transaction operation {} must have non-zero predicted cost",
                operation.name
            ),
        });
    }
    if operation.block_uses.is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "transaction operation {} must declare block uses",
                operation.name
            ),
        });
    }
    Ok(())
}

fn record_execution_decision(ledger: &mut TokenLedger, operation: &TransactionOperation) {
    ledger.record_execution_decision(ExecutionDecision {
        operation: operation.name,
        executor_selected: operation.executor,
        candidate_costs: vec![
            CandidateCost::estimated("selected", operation.predicted_visible_ns),
            CandidateCost::estimated("host-roundtrip", operation.predicted_visible_ns * 4),
        ],
        reason: "transaction critical-path plan",
        predicted_visible_ns: operation.predicted_visible_ns,
        actual_visible_ns: None,
        metric_source: MetricSource::EstimatedModel,
    });
}

fn record_operation_events(
    ledger: &mut TokenLedger,
    operation: &TransactionOperation,
    start_ns: u64,
) -> Result<()> {
    match operation.executor {
        ExecutionOwner::Gpu(device) => {
            if operation.graph_capturable {
                ledger.record(LedgerEvent {
                    kind: LedgerEventKind::GraphReplay,
                    sync_class: None,
                    metric_source: MetricSource::EstimatedModel,
                    block_id: None,
                    from_tier: None,
                    to_tier: None,
                    bytes: 0,
                    latency_ns: 1,
                    label: "transaction_graph_replay",
                });
            }
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::KernelLaunch,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: None,
                from_tier: None,
                to_tier: None,
                bytes: 0,
                latency_ns: 1,
                label: "transaction_kernel_launch",
            });
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::DeviceActivity,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: None,
                from_tier: None,
                to_tier: None,
                bytes: 0,
                latency_ns: operation.predicted_visible_ns,
                label: operation.name,
            });
            ledger.record_device_span(DeviceTimelineSpan::new(
                device,
                start_ns,
                start_ns.saturating_add(operation.predicted_visible_ns),
                MetricSource::EstimatedModel,
                operation.name,
            ))?;
            ledger.record_sync(
                SyncClass::HardSync,
                None,
                None,
                None,
                0,
                1,
                MetricSource::EstimatedModel,
                "transaction_device_dependency",
            );
        }
        ExecutionOwner::Cpu => {
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::CpuActivity,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: None,
                from_tier: None,
                to_tier: None,
                bytes: 0,
                latency_ns: operation.predicted_visible_ns,
                label: operation.name,
            });
            if operation.kind == TransactionOperationKind::HostObservation {
                ledger.record_sync(
                    SyncClass::SoftVisibilitySync,
                    None,
                    None,
                    None,
                    0,
                    operation.predicted_visible_ns,
                    MetricSource::EstimatedModel,
                    "transaction_host_observation",
                );
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_and_record_block_uses(
    ledger: &mut TokenLedger,
    registry: &BlockRegistry,
    operation: &TransactionOperation,
) -> Result<()> {
    for block_use in &operation.block_uses {
        let block =
            registry
                .block(block_use.block_id)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: format!(
                        "transaction operation {} references unknown block {}",
                        operation.name, block_use.block_id.0
                    ),
                })?;
        if block.tier != block_use.expected_tier {
            return Err(NervaError::ResidencyViolation {
                block_id: block.id,
                reason: format!(
                    "transaction block use '{}' expected {:?}, observed {:?}",
                    block_use.label, block_use.expected_tier, block.tier
                ),
            });
        }
        block.require_ready(block.authoritative_copy, block_use.required_version)?;
        ledger.record_block_version_dependency(BlockVersionDependency {
            block_id: block.id,
            required_version: block_use.required_version,
            observed_version: block.version,
            label: block_use.label,
        });
        if block_use.access.writes() && block.owner != block_use.owner {
            ledger.record_sync(
                SyncClass::PhaseHandoff,
                Some(block.id),
                Some(block.tier),
                Some(block_use.expected_tier),
                block.bytes,
                1,
                MetricSource::EstimatedModel,
                "transaction_phase_handoff",
            );
        }
    }
    Ok(())
}

fn summarize_transaction(
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
