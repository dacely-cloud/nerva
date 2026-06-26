use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::memory::tier::MemoryTier;

use crate::types::decision::{BlockVersionDependency, ExecutionDecision, ResidencyDecision};
use crate::types::event::{DeviceTimelineSpan, LedgerEvent, LedgerEventKind};
use crate::types::fallback::{FallbackClass, FallbackDecision};
use crate::types::metric::MetricSource;
use crate::types::sync::SyncClass;
use crate::types::token::timeline;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TokenLedger {
    pub token_index: u64,
    pub events: Vec<LedgerEvent>,
    pub device_timeline: Vec<DeviceTimelineSpan>,
    pub fallback_decisions: Vec<FallbackDecision>,
    pub block_version_dependencies: Vec<BlockVersionDependency>,
    pub residency_decisions: Vec<ResidencyDecision>,
    pub execution_decisions: Vec<ExecutionDecision>,
    pub hot_path_allocations: u64,
}

impl TokenLedger {
    pub fn new(token_index: u64) -> Self {
        Self {
            token_index,
            events: Vec::new(),
            device_timeline: Vec::new(),
            fallback_decisions: Vec::new(),
            block_version_dependencies: Vec::new(),
            residency_decisions: Vec::new(),
            execution_decisions: Vec::new(),
            hot_path_allocations: 0,
        }
    }

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

    pub fn require_satisfied_block_versions(&self) -> Result<()> {
        for dependency in &self.block_version_dependencies {
            if dependency.observed_version < dependency.required_version {
                return Err(NervaError::ResidencyViolation {
                    block_id: dependency.block_id,
                    reason: format!(
                        "block version dependency '{}' requires {}, observed {}",
                        dependency.label, dependency.required_version, dependency.observed_version
                    ),
                });
            }
        }
        Ok(())
    }

    pub fn device_active_ns(&self, device: DeviceOrdinal) -> Result<u64> {
        let (active_ns, _) = timeline::device_timeline_totals(&self.device_timeline, device)?;
        Ok(active_ns)
    }

    pub fn device_idle_ns(&self, device: DeviceOrdinal) -> Result<u64> {
        let (_, idle_ns) = timeline::device_timeline_totals(&self.device_timeline, device)?;
        Ok(idle_ns)
    }

    pub fn require_zero_hot_path_allocations(&self) -> Result<()> {
        if self.hot_path_allocations == 0 {
            Ok(())
        } else {
            Err(NervaError::InvalidArgument {
                reason: format!(
                    "hot path allocation counter is {}",
                    self.hot_path_allocations
                ),
            })
        }
    }

    pub fn require_classified_syncs(&self) -> Result<()> {
        for event in &self.events {
            match (event.kind, event.sync_class) {
                (LedgerEventKind::Sync, Some(_)) => {}
                (LedgerEventKind::Sync, None) => {
                    return Err(NervaError::InvalidArgument {
                        reason: format!("sync event '{}' is missing SyncClass", event.label),
                    });
                }
                (_, Some(_)) => {
                    return Err(NervaError::InvalidArgument {
                        reason: format!(
                            "non-sync event '{}' carries an invalid SyncClass",
                            event.label
                        ),
                    });
                }
                (_, None) => {}
            }
        }
        Ok(())
    }
}
