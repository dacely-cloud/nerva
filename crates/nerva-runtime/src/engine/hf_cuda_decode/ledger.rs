use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_cuda::block::forward::summary::CudaBlockForwardSummary;
use nerva_cuda::decode::hf_chain::summary::CudaHfDecodeChainSummary;
use nerva_cuda::decode::hf_step::summary::CudaHfDecodeStepSummary;
use nerva_cuda::sampler::hf_head::summary::CudaHfSamplerSummary;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

macro_rules! cuda_view {
    ($cuda:expr) => {
        CudaLedgerView {
            resident_weight_bytes: $cuda.resident_weight_bytes,
            h2d_bytes: $cuda.h2d_bytes,
            d2h_bytes: $cuda.d2h_bytes,
            sync_calls: $cuda.sync_calls,
        }
    };
}

pub(super) fn record_layer_execution(ledger: &mut TokenLedger, cuda: &CudaBlockForwardSummary) {
    record_cuda_execution(
        ledger,
        cuda_view!(cuda),
        "hf_cuda_seed_decode_layer",
        "cuda-loaded-hf-layer",
        "cpu-loaded-hf-layer",
        "loaded HF layer executed through CUDA block forward contract",
        "hf_cuda_seed_decode_kernel",
        "hf_cuda_seed_decode_layer",
        "hf_cuda_seed_decode_visibility",
    );
}

pub(super) fn record_sampler_execution(ledger: &mut TokenLedger, cuda: &CudaHfSamplerSummary) {
    record_cuda_execution(
        ledger,
        cuda_view!(cuda),
        "hf_cuda_final_head_sampler",
        "cuda-final-head-sampler",
        "cpu-final-head-sampler",
        "loaded HF final norm, LM head, and greedy argmax executed on CUDA",
        "hf_cuda_final_head_sampler_kernel",
        "hf_cuda_final_head_sampler",
        "hf_cuda_final_head_token_visibility",
    );
}

pub(super) fn record_fused_step_execution(
    ledger: &mut TokenLedger,
    cuda: &CudaHfDecodeStepSummary,
) {
    record_cuda_execution(
        ledger,
        cuda_view!(cuda),
        "hf_cuda_fused_decode_step",
        "cuda-fused-hf-decode-step",
        "split-layer-sampler-step",
        "loaded HF layer, final head, and greedy token executed in one CUDA step",
        "hf_cuda_fused_decode_step_kernel",
        "hf_cuda_fused_decode_step",
        "hf_cuda_fused_decode_token_visibility",
    );
}

pub(super) fn record_chain_execution(ledger: &mut TokenLedger, cuda: &CudaHfDecodeChainSummary) {
    record_cuda_execution(
        ledger,
        cuda_view!(cuda),
        "hf_cuda_fused_decode_chain",
        "cuda-fused-hf-decode-chain",
        "split-layer-sampler-chain",
        "loaded HF layer chain, final head, and greedy token executed in one CUDA step",
        "hf_cuda_fused_decode_chain_kernel",
        "hf_cuda_fused_decode_chain",
        "hf_cuda_fused_decode_chain_visibility",
    );
}

#[derive(Copy, Clone)]
struct CudaLedgerView {
    resident_weight_bytes: u64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    sync_calls: u64,
}

impl CudaLedgerView {
    fn visible_ns(self) -> u64 {
        (self.h2d_bytes + self.d2h_bytes + self.resident_weight_bytes).max(1)
    }
}

fn record_cuda_execution(
    ledger: &mut TokenLedger,
    view: CudaLedgerView,
    operation: &'static str,
    cuda_candidate: &'static str,
    cpu_candidate: &'static str,
    reason: &'static str,
    kernel_label: &'static str,
    activity_label: &'static str,
    sync_label: &'static str,
) {
    let visible_ns = view.visible_ns();
    ledger.record_execution_decision(ExecutionDecision {
        operation,
        executor_selected: ExecutionOwner::Gpu(DeviceOrdinal(0)),
        candidate_costs: vec![
            CandidateCost::estimated(cuda_candidate, visible_ns),
            CandidateCost::estimated(cpu_candidate, visible_ns.saturating_mul(2)),
        ],
        reason,
        predicted_visible_ns: visible_ns,
        actual_visible_ns: None,
        metric_source: MetricSource::EstimatedModel,
    });
    record_copy(
        ledger,
        MemoryTier::PinnedDram,
        MemoryTier::Vram,
        view.h2d_bytes,
    );
    record_copy(
        ledger,
        MemoryTier::Vram,
        MemoryTier::PinnedDram,
        view.d2h_bytes,
    );
    record_event(ledger, LedgerEventKind::KernelLaunch, 0, kernel_label);
    record_event(
        ledger,
        LedgerEventKind::DeviceActivity,
        visible_ns,
        activity_label,
    );
    ledger.record_sync(
        SyncClass::HardSync,
        None,
        Some(MemoryTier::Vram),
        Some(MemoryTier::PinnedDram),
        view.d2h_bytes as usize,
        view.sync_calls.max(1),
        MetricSource::EstimatedModel,
        sync_label,
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
