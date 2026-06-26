use crate::types::event::LedgerEventKind;
use crate::types::fallback::FallbackClass;
use crate::types::metric::MetricSource;
use crate::types::sync::SyncClass;
use crate::types::token::ledger::TokenLedger;

impl TokenLedger {
    pub fn total_latency_ns(&self) -> u64 {
        self.events.iter().map(|event| event.latency_ns).sum()
    }

    pub fn event_count(&self, kind: LedgerEventKind) -> u64 {
        self.events
            .iter()
            .filter(|event| event.kind == kind)
            .count() as u64
    }

    pub fn latency_ns_for(&self, kind: LedgerEventKind) -> u64 {
        self.events
            .iter()
            .filter(|event| event.kind == kind)
            .map(|event| event.latency_ns)
            .sum()
    }

    pub fn event_count_for_source(&self, source: MetricSource) -> u64 {
        self.events
            .iter()
            .filter(|event| event.metric_source == source)
            .count() as u64
    }

    pub fn latency_ns_for_source(&self, source: MetricSource) -> u64 {
        self.events
            .iter()
            .filter(|event| event.metric_source == source)
            .map(|event| event.latency_ns)
            .sum()
    }

    pub fn sync_count_for(&self, sync_class: SyncClass) -> u64 {
        self.events
            .iter()
            .filter(|event| {
                event.kind == LedgerEventKind::Sync && event.sync_class == Some(sync_class)
            })
            .count() as u64
    }

    pub fn sync_latency_ns_for(&self, sync_class: SyncClass) -> u64 {
        self.events
            .iter()
            .filter(|event| {
                event.kind == LedgerEventKind::Sync && event.sync_class == Some(sync_class)
            })
            .map(|event| event.latency_ns)
            .sum()
    }

    pub fn fallback_count(&self) -> u64 {
        self.fallback_decisions.len() as u64
    }

    pub fn fallback_count_for(&self, class: FallbackClass) -> u64 {
        self.fallback_decisions
            .iter()
            .filter(|decision| decision.class == class)
            .count() as u64
    }
}
