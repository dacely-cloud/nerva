use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_cuda::block::forward::summary::CudaBlockForwardSummary;
use nerva_cuda::decode::hf_step::summary::CudaHfDecodeStepSummary;
use nerva_cuda::sampler::hf_head::summary::CudaHfSamplerSummary;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

pub(super) fn record_layer_execution(ledger: &mut TokenLedger, cuda: &CudaBlockForwardSummary) {
    let visible_ns = (cuda.h2d_bytes + cuda.d2h_bytes + cuda.resident_weight_bytes).max(1);
    ledger.record_execution_decision(ExecutionDecision {
        operation: "hf_cuda_seed_decode_layer",
        executor_selected: ExecutionOwner::Gpu(DeviceOrdinal(0)),
        candidate_costs: vec![
            CandidateCost::estimated("cuda-loaded-hf-layer", visible_ns),
            CandidateCost::estimated("cpu-loaded-hf-layer", visible_ns.saturating_mul(2)),
        ],
        reason: "loaded HF layer executed through CUDA block forward contract",
        predicted_visible_ns: visible_ns,
        actual_visible_ns: None,
        metric_source: MetricSource::EstimatedModel,
    });
    record_copy(
        ledger,
        MemoryTier::PinnedDram,
        MemoryTier::Vram,
        cuda.h2d_bytes,
    );
    record_copy(
        ledger,
        MemoryTier::Vram,
        MemoryTier::PinnedDram,
        cuda.d2h_bytes,
    );
    record_event(
        ledger,
        LedgerEventKind::KernelLaunch,
        0,
        "hf_cuda_seed_decode_kernel",
    );
    record_event(
        ledger,
        LedgerEventKind::DeviceActivity,
        visible_ns,
        "hf_cuda_seed_decode_layer",
    );
    ledger.record_sync(
        SyncClass::HardSync,
        None,
        Some(MemoryTier::Vram),
        Some(MemoryTier::PinnedDram),
        cuda.d2h_bytes as usize,
        cuda.sync_calls.max(1),
        MetricSource::EstimatedModel,
        "hf_cuda_seed_decode_visibility",
    );
}

pub(super) fn record_sampler_execution(ledger: &mut TokenLedger, cuda: &CudaHfSamplerSummary) {
    let visible_ns = (cuda.h2d_bytes + cuda.d2h_bytes + cuda.resident_weight_bytes).max(1);
    ledger.record_execution_decision(ExecutionDecision {
        operation: "hf_cuda_final_head_sampler",
        executor_selected: ExecutionOwner::Gpu(DeviceOrdinal(0)),
        candidate_costs: vec![
            CandidateCost::estimated("cuda-final-head-sampler", visible_ns),
            CandidateCost::estimated("cpu-final-head-sampler", visible_ns.saturating_mul(2)),
        ],
        reason: "loaded HF final norm, LM head, and greedy argmax executed on CUDA",
        predicted_visible_ns: visible_ns,
        actual_visible_ns: None,
        metric_source: MetricSource::EstimatedModel,
    });
    record_copy(
        ledger,
        MemoryTier::PinnedDram,
        MemoryTier::Vram,
        cuda.h2d_bytes,
    );
    record_copy(
        ledger,
        MemoryTier::Vram,
        MemoryTier::PinnedDram,
        cuda.d2h_bytes,
    );
    record_event(
        ledger,
        LedgerEventKind::KernelLaunch,
        0,
        "hf_cuda_final_head_sampler_kernel",
    );
    record_event(
        ledger,
        LedgerEventKind::DeviceActivity,
        visible_ns,
        "hf_cuda_final_head_sampler",
    );
    ledger.record_sync(
        SyncClass::HardSync,
        None,
        Some(MemoryTier::Vram),
        Some(MemoryTier::PinnedDram),
        cuda.d2h_bytes as usize,
        cuda.sync_calls.max(1),
        MetricSource::EstimatedModel,
        "hf_cuda_final_head_token_visibility",
    );
}

pub(super) fn record_fused_step_execution(
    ledger: &mut TokenLedger,
    cuda: &CudaHfDecodeStepSummary,
) {
    let visible_ns = (cuda.h2d_bytes + cuda.d2h_bytes + cuda.resident_weight_bytes).max(1);
    ledger.record_execution_decision(ExecutionDecision {
        operation: "hf_cuda_fused_decode_step",
        executor_selected: ExecutionOwner::Gpu(DeviceOrdinal(0)),
        candidate_costs: vec![
            CandidateCost::estimated("cuda-fused-hf-decode-step", visible_ns),
            CandidateCost::estimated("split-layer-sampler-step", visible_ns.saturating_mul(2)),
        ],
        reason: "loaded HF layer, final head, and greedy token executed in one CUDA step",
        predicted_visible_ns: visible_ns,
        actual_visible_ns: None,
        metric_source: MetricSource::EstimatedModel,
    });
    record_copy(
        ledger,
        MemoryTier::PinnedDram,
        MemoryTier::Vram,
        cuda.h2d_bytes,
    );
    record_copy(
        ledger,
        MemoryTier::Vram,
        MemoryTier::PinnedDram,
        cuda.d2h_bytes,
    );
    record_event(
        ledger,
        LedgerEventKind::KernelLaunch,
        0,
        "hf_cuda_fused_decode_step_kernel",
    );
    record_event(
        ledger,
        LedgerEventKind::DeviceActivity,
        visible_ns,
        "hf_cuda_fused_decode_step",
    );
    ledger.record_sync(
        SyncClass::HardSync,
        None,
        Some(MemoryTier::Vram),
        Some(MemoryTier::PinnedDram),
        cuda.d2h_bytes as usize,
        cuda.sync_calls.max(1),
        MetricSource::EstimatedModel,
        "hf_cuda_fused_decode_token_visibility",
    );
}

fn record_copy(ledger: &mut TokenLedger, from: MemoryTier, to: MemoryTier, bytes: u64) {
    if bytes == 0 {
        return;
    }
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Copy,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(from),
        to_tier: Some(to),
        bytes: bytes as usize,
        latency_ns: bytes.max(1),
        label: "hf_cuda_seed_decode_copy",
    });
}

fn record_event(
    ledger: &mut TokenLedger,
    kind: LedgerEventKind,
    latency_ns: u64,
    label: &'static str,
) {
    ledger.record(LedgerEvent {
        kind,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(MemoryTier::Vram),
        to_tier: Some(MemoryTier::Vram),
        bytes: 0,
        latency_ns,
        label,
    });
}
