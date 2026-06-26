#![forbid(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use nerva_core::{CostSource, ExecutionOwner, MemoryTier, NervaError, ResidentBlockId, Result};

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerEvent {
    pub kind: LedgerEventKind,
    pub block_id: Option<ResidentBlockId>,
    pub from_tier: Option<MemoryTier>,
    pub to_tier: Option<MemoryTier>,
    pub bytes: usize,
    pub latency_ns: u64,
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
    pub residency_decisions: Vec<ResidencyDecision>,
    pub execution_decisions: Vec<ExecutionDecision>,
    pub hot_path_allocations: u64,
}

impl TokenLedger {
    pub fn new(token_index: u64) -> Self {
        Self {
            token_index,
            events: Vec::new(),
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

    pub fn record_residency_decision(&mut self, decision: ResidencyDecision) {
        self.residency_decisions.push(decision);
    }

    pub fn record_execution_decision(&mut self, decision: ExecutionDecision) {
        self.execution_decisions.push(decision);
    }

    pub fn record_hot_path_allocation_attempt(
        &mut self,
        label: &'static str,
        bytes: usize,
        to_tier: MemoryTier,
    ) {
        self.record(LedgerEvent {
            kind: LedgerEventKind::Allocation,
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
            block_id: None,
            from_tier: None,
            to_tier: Some(MemoryTier::Vram),
            bytes: 0,
            latency_ns: 2,
            label: "graph",
        });
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::DeviceActivity,
            block_id: None,
            from_tier: None,
            to_tier: Some(MemoryTier::Vram),
            bytes: 0,
            latency_ns: 7,
            label: "device",
        });
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Sync,
            block_id: None,
            from_tier: Some(MemoryTier::Vram),
            to_tier: Some(MemoryTier::PinnedDram),
            bytes: 0,
            latency_ns: 3,
            label: "soft_visibility_host_wait",
        });

        assert_eq!(ledger.event_count(LedgerEventKind::GraphReplay), 1);
        assert_eq!(ledger.event_count(LedgerEventKind::DeviceActivity), 1);
        assert_eq!(ledger.event_count(LedgerEventKind::Sync), 1);
        assert_eq!(ledger.latency_ns_for(LedgerEventKind::DeviceActivity), 7);
        assert_eq!(ledger.latency_ns_for(LedgerEventKind::Sync), 3);
        assert_eq!(ledger.total_latency_ns(), 12);
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
