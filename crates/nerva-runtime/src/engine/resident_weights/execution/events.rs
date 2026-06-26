use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::fallback::{FallbackClass, FallbackDecision};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::resident_weights::helpers::{
    div_ceil_u64, estimate_cpu_dram_weight_ns, estimate_gpu_resident_weight_ns,
};
use crate::weights::execution::step::ResidentWeightExecutionStep;

pub(super) fn record_cpu_dram_step(ledger: &mut TokenLedger, step: &ResidentWeightExecutionStep) {
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::CpuActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(step.block_id),
        from_tier: Some(MemoryTier::Dram),
        to_tier: Some(MemoryTier::Dram),
        bytes: step.bytes,
        latency_ns: step.predicted_visible_ns,
        label: "resident_weight_cpu_dram_matvec",
    });
}

pub(super) fn record_gpu_resident_step(
    ledger: &mut TokenLedger,
    step: &ResidentWeightExecutionStep,
) {
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::DeviceActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(step.block_id),
        from_tier: Some(MemoryTier::Vram),
        to_tier: Some(MemoryTier::Vram),
        bytes: step.bytes,
        latency_ns: step.predicted_visible_ns,
        label: "resident_weight_gpu_matvec",
    });
}

pub(super) fn record_gpu_staged_step(
    ledger: &mut TokenLedger,
    step: &ResidentWeightExecutionStep,
    source_tier: MemoryTier,
) {
    let copy_ns = div_ceil_u64(step.bytes as u64, 24);
    let compute_ns = estimate_gpu_resident_weight_ns(step.bytes);
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Copy,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(step.block_id),
        from_tier: Some(source_tier),
        to_tier: Some(MemoryTier::Vram),
        bytes: step.bytes,
        latency_ns: copy_ns,
        label: "resident_weight_stage_to_gpu",
    });
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::DeviceActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(step.block_id),
        from_tier: Some(MemoryTier::Vram),
        to_tier: Some(MemoryTier::Vram),
        bytes: step.bytes,
        latency_ns: compute_ns,
        label: "resident_weight_gpu_staged_matvec",
    });
}

pub(super) fn record_cpu_exact_fallback_step(
    ledger: &mut TokenLedger,
    step: &ResidentWeightExecutionStep,
    source_tier: MemoryTier,
) {
    ledger.record_fallback_decision(FallbackDecision {
        label: "resident_weight_exact_cpu_fallback_run",
        class: FallbackClass::ExactNamed,
        requested: "cuda_dense_matvec",
        selected: step.kernel_name,
        reason: "executing declared exact CPU fallback step",
        visible_ns: Some(step.predicted_visible_ns),
        metric_source: MetricSource::EstimatedModel,
    });
    if source_tier == MemoryTier::Vram || source_tier == MemoryTier::SharedHbmOrLpddr {
        let copy_ns = div_ceil_u64(step.bytes as u64, 24);
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Copy,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: Some(step.block_id),
            from_tier: Some(source_tier),
            to_tier: Some(MemoryTier::Dram),
            bytes: step.bytes,
            latency_ns: copy_ns,
            label: "resident_weight_fallback_to_cpu",
        });
    }
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::CpuActivity,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(step.block_id),
        from_tier: Some(MemoryTier::Dram),
        to_tier: Some(MemoryTier::Dram),
        bytes: step.bytes,
        latency_ns: estimate_cpu_dram_weight_ns(step.bytes),
        label: "resident_weight_cpu_exact_fallback",
    });
}
