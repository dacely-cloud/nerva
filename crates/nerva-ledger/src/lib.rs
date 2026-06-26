#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use nerva_core::{
    CostSource, DeviceOrdinal, ExecutionOwner, MemoryTier, NervaError, ResidentBlockId, Result,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LedgerEventKind {
    GraphReplay,
    KernelLaunch,
    CpuActivity,
    DeviceActivity,
    Copy,
    Sync,
    Allocation,
    Eviction,
    Prefetch,
    Stall,
    Transport,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MetricSource {
    RuntimeTimestamp,
    GpuEvent,
    HardwareCounter,
    Profiler,
    TransportCompletion,
    EstimatedModel,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SyncClass {
    HardSync,
    SoftVisibilitySync,
    PolicySync,
    PhaseHandoff,
    DebugSync,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FallbackClass {
    ExactNamed,
    CapabilityDegraded,
    PolicySelected,
    DebugOnly,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerEvent {
    pub kind: LedgerEventKind,
    pub sync_class: Option<SyncClass>,
    pub metric_source: MetricSource,
    pub block_id: Option<ResidentBlockId>,
    pub from_tier: Option<MemoryTier>,
    pub to_tier: Option<MemoryTier>,
    pub bytes: usize,
    pub latency_ns: u64,
    pub label: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceTimelineSpan {
    pub device: DeviceOrdinal,
    pub start_ns: u64,
    pub end_ns: u64,
    pub metric_source: MetricSource,
    pub label: &'static str,
}

impl DeviceTimelineSpan {
    pub const fn new(
        device: DeviceOrdinal,
        start_ns: u64,
        end_ns: u64,
        metric_source: MetricSource,
        label: &'static str,
    ) -> Self {
        Self {
            device,
            start_ns,
            end_ns,
            metric_source,
            label,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FallbackDecision {
    pub label: &'static str,
    pub class: FallbackClass,
    pub requested: &'static str,
    pub selected: &'static str,
    pub reason: &'static str,
    pub visible_ns: Option<u64>,
    pub metric_source: MetricSource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockVersionDependency {
    pub block_id: ResidentBlockId,
    pub required_version: u64,
    pub observed_version: u64,
    pub label: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateCost {
    pub label: &'static str,
    pub visible_ns: Option<u64>,
    pub source: CostSource,
}

impl CandidateCost {
    pub const fn estimated(label: &'static str, visible_ns: u64) -> Self {
        Self {
            label,
            visible_ns: Some(visible_ns),
            source: CostSource::Estimated,
        }
    }

    pub const fn measured(label: &'static str, visible_ns: u64) -> Self {
        Self {
            label,
            visible_ns: Some(visible_ns),
            source: CostSource::Measured,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidencyDecision {
    pub block_id: ResidentBlockId,
    pub old_tier: MemoryTier,
    pub new_tier: MemoryTier,
    pub executor_selected: ExecutionOwner,
    pub candidate_costs: Vec<CandidateCost>,
    pub reason: &'static str,
    pub predicted_overlap_ns: u64,
    pub actual_visible_ns: Option<u64>,
    pub metric_source: MetricSource,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionDecision {
    pub operation: &'static str,
    pub executor_selected: ExecutionOwner,
    pub candidate_costs: Vec<CandidateCost>,
    pub reason: &'static str,
    pub predicted_visible_ns: u64,
    pub actual_visible_ns: Option<u64>,
    pub metric_source: MetricSource,
}

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

    pub fn record_sync(
        &mut self,
        sync_class: SyncClass,
        block_id: Option<ResidentBlockId>,
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
        validate_device_span(&span)?;
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
        let (active_ns, _) = self.device_timeline_totals(device)?;
        Ok(active_ns)
    }

    pub fn device_idle_ns(&self, device: DeviceOrdinal) -> Result<u64> {
        let (_, idle_ns) = self.device_timeline_totals(device)?;
        Ok(idle_ns)
    }

    fn device_timeline_totals(&self, device: DeviceOrdinal) -> Result<(u64, u64)> {
        let mut spans = self
            .device_timeline
            .iter()
            .filter(|span| span.device == device)
            .collect::<Vec<_>>();
        spans.sort_by_key(|span| (span.start_ns, span.end_ns));

        let mut active_ns = 0u64;
        let mut idle_ns = 0u64;
        let mut merged_end: Option<u64> = None;

        for span in spans {
            validate_device_span(span)?;
            match merged_end {
                None => {
                    active_ns = active_ns.saturating_add(span.end_ns - span.start_ns);
                    merged_end = Some(span.end_ns);
                }
                Some(end) if span.end_ns <= end => {}
                Some(end) if span.start_ns <= end => {
                    active_ns = active_ns.saturating_add(span.end_ns - end);
                    merged_end = Some(span.end_ns);
                }
                Some(end) => {
                    idle_ns = idle_ns.saturating_add(span.start_ns - end);
                    active_ns = active_ns.saturating_add(span.end_ns - span.start_ns);
                    merged_end = Some(span.end_ns);
                }
            }
        }

        Ok((active_ns, idle_ns))
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

fn validate_device_span(span: &DeviceTimelineSpan) -> Result<()> {
    if span.end_ns < span.start_ns {
        Err(NervaError::InvalidArgument {
            reason: format!("device span '{}' ends before it starts", span.label),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_events_increment_hot_path_count() {
        let mut ledger = TokenLedger::new(0);
        ledger.record_hot_path_allocation_attempt("test", 64, MemoryTier::Vram);
        assert_eq!(ledger.hot_path_allocations, 1);
        assert_eq!(ledger.total_latency_ns(), 0);
        assert_eq!(ledger.event_count(LedgerEventKind::Allocation), 1);
        assert!(ledger.require_zero_hot_path_allocations().is_err());
    }

    #[test]
    fn ledger_keeps_host_wait_and_device_activity_separate() {
        let mut ledger = TokenLedger::new(5);
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::GraphReplay,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: None,
            from_tier: None,
            to_tier: Some(MemoryTier::Vram),
            bytes: 0,
            latency_ns: 2,
            label: "graph",
        });
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::DeviceActivity,
            sync_class: None,
            metric_source: MetricSource::GpuEvent,
            block_id: None,
            from_tier: None,
            to_tier: Some(MemoryTier::Vram),
            bytes: 0,
            latency_ns: 7,
            label: "device",
        });
        ledger.record_sync(
            SyncClass::SoftVisibilitySync,
            None,
            Some(MemoryTier::Vram),
            Some(MemoryTier::PinnedDram),
            0,
            3,
            MetricSource::EstimatedModel,
            "soft_visibility_host_wait",
        );

        assert_eq!(ledger.event_count(LedgerEventKind::GraphReplay), 1);
        assert_eq!(ledger.event_count(LedgerEventKind::DeviceActivity), 1);
        assert_eq!(ledger.event_count(LedgerEventKind::Sync), 1);
        assert_eq!(ledger.sync_count_for(SyncClass::SoftVisibilitySync), 1);
        assert_eq!(ledger.latency_ns_for(LedgerEventKind::DeviceActivity), 7);
        assert_eq!(ledger.latency_ns_for(LedgerEventKind::Sync), 3);
        assert_eq!(ledger.sync_latency_ns_for(SyncClass::SoftVisibilitySync), 3);
        assert_eq!(ledger.event_count_for_source(MetricSource::GpuEvent), 1);
        assert_eq!(
            ledger.latency_ns_for_source(MetricSource::EstimatedModel),
            5
        );
        ledger
            .record_device_span(DeviceTimelineSpan::new(
                DeviceOrdinal(0),
                0,
                7,
                MetricSource::GpuEvent,
                "device_active",
            ))
            .unwrap();
        assert_eq!(ledger.device_active_ns(DeviceOrdinal(0)).unwrap(), 7);
        assert_eq!(ledger.device_idle_ns(DeviceOrdinal(0)).unwrap(), 0);
        assert_eq!(ledger.total_latency_ns(), 12);
        assert!(ledger.require_classified_syncs().is_ok());
    }

    #[test]
    fn device_idle_is_derived_from_device_timeline_gaps() {
        let mut ledger = TokenLedger::new(0);
        ledger
            .record_device_span(DeviceTimelineSpan::new(
                DeviceOrdinal(0),
                0,
                10,
                MetricSource::GpuEvent,
                "kernel_a",
            ))
            .unwrap();
        ledger
            .record_device_span(DeviceTimelineSpan::new(
                DeviceOrdinal(0),
                15,
                25,
                MetricSource::GpuEvent,
                "kernel_b",
            ))
            .unwrap();
        ledger
            .record_device_span(DeviceTimelineSpan::new(
                DeviceOrdinal(0),
                20,
                30,
                MetricSource::GpuEvent,
                "overlap_kernel",
            ))
            .unwrap();
        ledger.record_sync(
            SyncClass::SoftVisibilitySync,
            None,
            Some(MemoryTier::Vram),
            Some(MemoryTier::PinnedDram),
            0,
            100,
            MetricSource::RuntimeTimestamp,
            "host_wait_not_gpu_idle",
        );

        assert_eq!(ledger.latency_ns_for(LedgerEventKind::Sync), 100);
        assert_eq!(ledger.device_active_ns(DeviceOrdinal(0)).unwrap(), 25);
        assert_eq!(ledger.device_idle_ns(DeviceOrdinal(0)).unwrap(), 5);
    }

    #[test]
    fn device_timeline_rejects_invalid_spans() {
        let mut ledger = TokenLedger::new(0);
        let result = ledger.record_device_span(DeviceTimelineSpan::new(
            DeviceOrdinal(0),
            10,
            9,
            MetricSource::GpuEvent,
            "bad_span",
        ));

        assert!(result.is_err());
    }

    #[test]
    fn fallback_decisions_are_recorded_separately_from_events() {
        let mut ledger = TokenLedger::new(0);
        ledger.record_fallback_decision(FallbackDecision {
            label: "cpu_reference_fallback",
            class: FallbackClass::ExactNamed,
            requested: "cuda_dense_matvec_f16",
            selected: "cpu_reference_dense_matvec_f32",
            reason: "declared exact fallback",
            visible_ns: Some(12),
            metric_source: MetricSource::EstimatedModel,
        });
        ledger.record_fallback_decision(FallbackDecision {
            label: "host_staged_transport",
            class: FallbackClass::CapabilityDegraded,
            requested: "gpu_direct_rdma",
            selected: "pinned_host_bounce",
            reason: "direct path unverified",
            visible_ns: Some(7),
            metric_source: MetricSource::EstimatedModel,
        });

        assert_eq!(ledger.events.len(), 0);
        assert_eq!(ledger.fallback_count(), 2);
        assert_eq!(ledger.fallback_count_for(FallbackClass::ExactNamed), 1);
        assert_eq!(
            ledger.fallback_count_for(FallbackClass::CapabilityDegraded),
            1
        );
    }

    #[test]
    fn block_version_dependencies_validate_observed_versions() {
        let mut ledger = TokenLedger::new(0);
        ledger.record_block_version_dependency(BlockVersionDependency {
            block_id: ResidentBlockId(7),
            required_version: 2,
            observed_version: 2,
            label: "weight_step",
        });
        ledger.record_block_version_dependency(BlockVersionDependency {
            block_id: ResidentBlockId(8),
            required_version: 2,
            observed_version: 3,
            label: "newer_replica",
        });

        assert_eq!(ledger.block_version_dependencies.len(), 2);
        assert!(ledger.require_satisfied_block_versions().is_ok());
    }

    #[test]
    fn block_version_dependencies_reject_stale_observations() {
        let mut ledger = TokenLedger::new(0);
        ledger.record_block_version_dependency(BlockVersionDependency {
            block_id: ResidentBlockId(7),
            required_version: 4,
            observed_version: 3,
            label: "stale_weight_step",
        });

        assert!(ledger.require_satisfied_block_versions().is_err());
    }

    #[test]
    fn classified_sync_validation_rejects_missing_or_misplaced_classes() {
        let mut missing = TokenLedger::new(0);
        missing.record(LedgerEvent {
            kind: LedgerEventKind::Sync,
            sync_class: None,
            metric_source: MetricSource::RuntimeTimestamp,
            block_id: None,
            from_tier: None,
            to_tier: None,
            bytes: 0,
            latency_ns: 1,
            label: "unclassified_wait",
        });
        assert!(missing.require_classified_syncs().is_err());

        let mut misplaced = TokenLedger::new(1);
        misplaced.record(LedgerEvent {
            kind: LedgerEventKind::Copy,
            sync_class: Some(SyncClass::HardSync),
            metric_source: MetricSource::RuntimeTimestamp,
            block_id: None,
            from_tier: Some(MemoryTier::Dram),
            to_tier: Some(MemoryTier::Vram),
            bytes: 4,
            latency_ns: 1,
            label: "copy_with_sync_class",
        });
        assert!(misplaced.require_classified_syncs().is_err());
    }

    #[test]
    fn residency_decisions_are_recorded_separately_from_timing_events() {
        let mut ledger = TokenLedger::new(3);
        ledger.record_residency_decision(ResidencyDecision {
            block_id: ResidentBlockId(9),
            old_tier: MemoryTier::Dram,
            new_tier: MemoryTier::Vram,
            executor_selected: ExecutionOwner::Cpu,
            candidate_costs: vec![
                CandidateCost::estimated("cpu-dram", 100),
                CandidateCost::estimated("gpu-prefetch", 80),
            ],
            reason: "prefetch hides transfer",
            predicted_overlap_ns: 40,
            actual_visible_ns: None,
            metric_source: MetricSource::EstimatedModel,
        });

        assert_eq!(ledger.events.len(), 0);
        assert_eq!(ledger.residency_decisions.len(), 1);
        assert_eq!(
            ledger.residency_decisions[0].candidate_costs[1].label,
            "gpu-prefetch"
        );
        assert!(ledger.require_zero_hot_path_allocations().is_ok());
    }

    #[test]
    fn execution_decisions_record_operation_placement() {
        let mut ledger = TokenLedger::new(8);
        ledger.record_execution_decision(ExecutionDecision {
            operation: "matvec",
            executor_selected: ExecutionOwner::Cpu,
            candidate_costs: vec![
                CandidateCost::estimated("cpu-dram", 16),
                CandidateCost::estimated("gpu-staged", 68),
            ],
            reason: "compute near warm DRAM weights",
            predicted_visible_ns: 16,
            actual_visible_ns: Some(16),
            metric_source: MetricSource::EstimatedModel,
        });

        assert_eq!(ledger.events.len(), 0);
        assert_eq!(ledger.execution_decisions.len(), 1);
        assert_eq!(ledger.execution_decisions[0].operation, "matvec");
        assert_eq!(
            ledger.execution_decisions[0].candidate_costs[0].label,
            "cpu-dram"
        );
        assert!(ledger.require_zero_hot_path_allocations().is_ok());
    }
}
