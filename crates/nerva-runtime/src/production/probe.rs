use nerva_core::types::error::Result;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::fallback::{FallbackClass, FallbackDecision};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::production::summary::{ProductionInvariantStatus, ProductionInvariantSummary};

pub fn run_production_invariant_probe() -> Result<ProductionInvariantSummary> {
    let clean = clean_production_ledger();
    clean.require_production_runtime_invariants()?;
    clean.require_zero_hot_path_allocations()?;

    Ok(ProductionInvariantSummary {
        status: ProductionInvariantStatus::Ok,
        accepted_ledgers: 1,
        classified_sync_ledgers: u64::from(clean.require_classified_syncs().is_ok()),
        measured_fallbacks: clean.fallback_count(),
        debug_sync_rejections: rejection_count(debug_sync_ledger()),
        debug_fallback_rejections: rejection_count(debug_fallback_ledger()),
        unmeasured_fallback_rejections: rejection_count(unmeasured_fallback_ledger()),
        unnamed_fallback_rejections: rejection_count(unnamed_fallback_ledger()),
        hot_path_allocations: clean.hot_path_allocations,
        error: None,
    })
}

fn clean_production_ledger() -> TokenLedger {
    let mut ledger = TokenLedger::new(0);
    ledger.record_sync(
        SyncClass::HardSync,
        None,
        Some(MemoryTier::Vram),
        Some(MemoryTier::Vram),
        128,
        7,
        MetricSource::EstimatedModel,
        "production_decode_hard_sync",
    );
    ledger.record_fallback_decision(FallbackDecision {
        label: "production_exact_cpu_fallback",
        class: FallbackClass::ExactNamed,
        requested: "cuda_dense_matvec_f32",
        selected: "cpu_reference_dense_matvec_f32",
        reason: "declared exact kernel fallback",
        visible_ns: Some(100),
        metric_source: MetricSource::EstimatedModel,
    });
    ledger.record_fallback_decision(FallbackDecision {
        label: "production_pinned_host_degradation",
        class: FallbackClass::CapabilityDegraded,
        requested: "gpu_direct_rdma",
        selected: "pinned_host_bounce",
        reason: "direct path unsupported by capability probe",
        visible_ns: Some(250),
        metric_source: MetricSource::EstimatedModel,
    });
    ledger
}

fn debug_sync_ledger() -> TokenLedger {
    let mut ledger = TokenLedger::new(1);
    ledger.record_sync(
        SyncClass::DebugSync,
        None,
        None,
        None,
        0,
        1,
        MetricSource::RuntimeTimestamp,
        "debug_device_synchronize",
    );
    ledger
}

fn debug_fallback_ledger() -> TokenLedger {
    let mut ledger = TokenLedger::new(2);
    ledger.record_fallback_decision(FallbackDecision {
        label: "debug_framework_fallback",
        class: FallbackClass::DebugOnly,
        requested: "cuda_graph_replay",
        selected: "debug_host_loop",
        reason: "debug-only path",
        visible_ns: Some(1),
        metric_source: MetricSource::EstimatedModel,
    });
    ledger
}

fn unmeasured_fallback_ledger() -> TokenLedger {
    let mut ledger = TokenLedger::new(3);
    ledger.record_fallback_decision(FallbackDecision {
        label: "unmeasured_transport_degradation",
        class: FallbackClass::CapabilityDegraded,
        requested: "gpu_direct_rdma",
        selected: "pinned_host_bounce",
        reason: "direct path unsupported",
        visible_ns: None,
        metric_source: MetricSource::EstimatedModel,
    });
    ledger
}

fn unnamed_fallback_ledger() -> TokenLedger {
    let mut ledger = TokenLedger::new(4);
    ledger.record_fallback_decision(FallbackDecision {
        label: "",
        class: FallbackClass::ExactNamed,
        requested: "cuda_dense_matvec",
        selected: "cpu_reference_dense_matvec",
        reason: "declared fallback",
        visible_ns: Some(10),
        metric_source: MetricSource::EstimatedModel,
    });
    ledger
}

fn rejection_count(ledger: TokenLedger) -> u64 {
    u64::from(ledger.require_production_runtime_invariants().is_err())
}
