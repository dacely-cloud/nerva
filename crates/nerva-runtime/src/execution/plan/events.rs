use nerva_core::types::error::Result;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::event::{DeviceTimelineSpan, LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::TokenLedger;

use crate::execution::types::{TransactionOperation, TransactionOperationKind};

pub fn record_operation_events(
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
