use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ResidencyDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::kv::residency::types::{KvResidencyAction, KvResidencyPlan, KvResidencyPlanEntry};

impl KvResidencyPlan {
    pub fn record_decisions_to_ledger(&self, ledger: &mut TokenLedger) {
        for entry in &self.entries {
            ledger.record_residency_decision(ResidencyDecision {
                block_id: entry.block_id,
                old_tier: entry.old_tier,
                new_tier: entry.new_tier,
                executor_selected: ExecutionOwner::Gpu(DeviceOrdinal(0)),
                candidate_costs: vec![
                    CandidateCost::estimated("keep-current-tier", 0),
                    CandidateCost::estimated("planned-tier", entry.predicted_visible_ns),
                ],
                reason: entry.reason,
                predicted_overlap_ns: 0,
                actual_visible_ns: None,
                metric_source: MetricSource::EstimatedModel,
            });
        }
    }

    pub fn record_events_to_ledger(&self, ledger: &mut TokenLedger) {
        for entry in &self.entries {
            match entry.action {
                KvResidencyAction::PrefetchToHot => {
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Prefetch,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_prefetch_scheduled",
                    });
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Copy,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_prefetch_copy",
                    });
                    record_visible_transfer_stall(ledger, entry);
                }
                KvResidencyAction::DemoteToWarm => {
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Eviction,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_demote_scheduled",
                    });
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Copy,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_demote_copy",
                    });
                    record_visible_transfer_stall(ledger, entry);
                }
                KvResidencyAction::EvictCold => {
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Eviction,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_cold_eviction",
                    });
                    if entry.changes_tier() {
                        ledger.record(LedgerEvent {
                            kind: LedgerEventKind::Copy,
                            sync_class: None,
                            metric_source: MetricSource::EstimatedModel,
                            block_id: Some(entry.block_id),
                            from_tier: Some(entry.old_tier),
                            to_tier: Some(entry.new_tier),
                            bytes: entry.bytes,
                            latency_ns: 0,
                            label: "kv_eviction_copy",
                        });
                    }
                    record_visible_transfer_stall(ledger, entry);
                }
                KvResidencyAction::KeepHot | KvResidencyAction::KeepWarm => {}
            }
        }
    }

    pub fn record_to_ledger(&self, ledger: &mut TokenLedger) {
        self.record_decisions_to_ledger(ledger);
        self.record_events_to_ledger(ledger);
    }
}

fn record_visible_transfer_stall(ledger: &mut TokenLedger, entry: &KvResidencyPlanEntry) {
    if entry.predicted_visible_ns == 0 {
        return;
    }
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Stall,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(entry.block_id),
        from_tier: Some(entry.old_tier),
        to_tier: Some(entry.new_tier),
        bytes: entry.bytes,
        latency_ns: entry.predicted_visible_ns,
        label: "kv_visible_transfer_stall",
    });
}
