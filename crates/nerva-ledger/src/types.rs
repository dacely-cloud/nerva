use nerva_core::types::{
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

impl LedgerEventKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GraphReplay => "graph_replay",
            Self::KernelLaunch => "kernel_launch",
            Self::CpuActivity => "cpu_activity",
            Self::DeviceActivity => "device_activity",
            Self::Copy => "copy",
            Self::Sync => "sync",
            Self::Allocation => "allocation",
            Self::Eviction => "eviction",
            Self::Prefetch => "prefetch",
            Self::Stall => "stall",
            Self::Transport => "transport",
        }
    }
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

impl MetricSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RuntimeTimestamp => "runtime_timestamp",
            Self::GpuEvent => "gpu_event",
            Self::HardwareCounter => "hardware_counter",
            Self::Profiler => "profiler",
            Self::TransportCompletion => "transport_completion",
            Self::EstimatedModel => "estimated_model",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SyncClass {
    HardSync,
    SoftVisibilitySync,
    PolicySync,
    PhaseHandoff,
    DebugSync,
}

impl SyncClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HardSync => "hard_sync",
            Self::SoftVisibilitySync => "soft_visibility_sync",
            Self::PolicySync => "policy_sync",
            Self::PhaseHandoff => "phase_handoff",
            Self::DebugSync => "debug_sync",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FallbackClass {
    ExactNamed,
    CapabilityDegraded,
    PolicySelected,
    DebugOnly,
}

impl FallbackClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExactNamed => "exact_named",
            Self::CapabilityDegraded => "capability_degraded",
            Self::PolicySelected => "policy_selected",
            Self::DebugOnly => "debug_only",
        }
    }
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
