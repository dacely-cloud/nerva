use nerva_core::types::error::Result;
use nerva_core::types::memory::tier::MemoryTier;

use crate::types::decision::{BlockVersionDependency, ExecutionDecision, ResidencyDecision};
use crate::types::event::{DeviceTimelineSpan, LedgerEvent, LedgerEventKind};
use crate::types::fallback::FallbackDecision;
use crate::types::metric::MetricSource;
use crate::types::sync::SyncClass;
use crate::types::token::ledger::TokenLedger;
use crate::types::token::timeline;

impl TokenLedger {
    pub fn record(&mut self, event: LedgerEvent) {
        if event.kind == LedgerEventKind::Allocation {
            self.hot_path_allocations += 1;
        }
        self.events.push(event);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_sync(
        &mut self,
        sync_class: SyncClass,
        block_id: Option<nerva_core::types::id::block::ResidentBlockId>,
        from_tier: Option<MemoryTier>,
        to_tier: Option<MemoryTier>,
        bytes: usize,
        latency_ns: u64,
        metric_source: MetricSource,
        label: &'static str,
    ) {
        self.record(LedgerEvent {
            kind: LedgerEventKind::Sync,
            sync_class: Some(sync_class),
            metric_source,
            block_id,
            from_tier,
            to_tier,
            bytes,
            latency_ns,
            label,
        });
    }

    pub fn record_residency_decision(&mut self, decision: ResidencyDecision) {
        self.residency_decisions.push(decision);
    }

    pub fn record_execution_decision(&mut self, decision: ExecutionDecision) {
        self.execution_decisions.push(decision);
    }

    pub fn record_fallback_decision(&mut self, decision: FallbackDecision) {
        self.fallback_decisions.push(decision);
    }

    pub fn record_block_version_dependency(&mut self, dependency: BlockVersionDependency) {
        self.block_version_dependencies.push(dependency);
    }

    pub fn record_device_span(&mut self, span: DeviceTimelineSpan) -> Result<()> {
        timeline::validate_device_span(&span)?;
        self.device_timeline.push(span);
        Ok(())
    }

    pub fn record_hot_path_allocation_attempt(
        &mut self,
        label: &'static str,
        bytes: usize,
        to_tier: MemoryTier,
    ) {
        self.record(LedgerEvent {
            kind: LedgerEventKind::Allocation,
            sync_class: None,
            metric_source: MetricSource::RuntimeTimestamp,
            block_id: None,
            from_tier: None,
            to_tier: Some(to_tier),
            bytes,
            latency_ns: 0,
            label,
        });
    }
}
