use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;

use crate::types::decision::{CandidateCost, ExecutionDecision, ResidencyDecision};
use crate::types::fallback::{FallbackClass, FallbackDecision};
use crate::types::metric::MetricSource;
use crate::types::token::ledger::TokenLedger;

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
    assert!(ledger.require_production_runtime_invariants().is_ok());
}

#[test]
fn production_runtime_invariants_reject_debug_or_unmeasured_fallbacks() {
    let mut debug = TokenLedger::new(1);
    debug.record_fallback_decision(FallbackDecision {
        label: "debug_framework_fallback",
        class: FallbackClass::DebugOnly,
        requested: "cuda_kernel",
        selected: "debug_cpu_path",
        reason: "debug probe",
        visible_ns: Some(1),
        metric_source: MetricSource::EstimatedModel,
    });
    assert!(debug.require_production_runtime_invariants().is_err());

    let mut unmeasured = TokenLedger::new(2);
    unmeasured.record_fallback_decision(FallbackDecision {
        label: "unmeasured_transport_fallback",
        class: FallbackClass::CapabilityDegraded,
        requested: "gpu_direct_rdma",
        selected: "pinned_host_bounce",
        reason: "direct path unsupported",
        visible_ns: None,
        metric_source: MetricSource::EstimatedModel,
    });
    assert!(unmeasured.require_production_runtime_invariants().is_err());
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
