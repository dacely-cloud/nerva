use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;
use nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use nerva_ledger::types::decision::{CandidateCost, ExecutionDecision};
use nerva_ledger::types::event::{DeviceTimelineSpan, LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

pub(super) fn sequence_ledgers(summary: &CudaHfDecodeSequenceSummary) -> Vec<TokenLedger> {
    let mut ledgers = Vec::with_capacity(summary.tokens.len());
    let last = summary.tokens.len().saturating_sub(1);
    for step in 0..summary.tokens.len() {
        let mut ledger = TokenLedger::new(step as u64);
        record_decision(&mut ledger, summary, step);
        if step == 0 {
            record_bootstrap_copies(&mut ledger, summary);
        }
        record_event(
            &mut ledger,
            LedgerEventKind::GraphReplay,
            graph_replay_ns(summary),
            MetricSource::EstimatedModel,
            "hf_cuda_sequence_graph_replay",
        );
        record_event(
            &mut ledger,
            LedgerEventKind::KernelLaunch,
            0,
            MetricSource::EstimatedModel,
            "hf_cuda_sequence_kernel",
        );
        let device_active_ns = device_active_ns(summary);
        let device_source = device_metric_source(summary);
        record_event(
            &mut ledger,
            LedgerEventKind::DeviceActivity,
            device_active_ns,
            device_source,
            "hf_cuda_sequence_device_step",
        );
        record_device_span(&mut ledger, device_active_ns, device_source);
        if step == last {
            record_copy(
                &mut ledger,
                false,
                summary.d2h_bytes,
                "hf_cuda_sequence_token_ring_d2h",
            );
            record_final_sync(&mut ledger, summary);
        }
        ledgers.push(ledger);
    }
    ledgers
}

fn record_bootstrap_copies(ledger: &mut TokenLedger, summary: &CudaHfDecodeSequenceSummary) {
    let descriptor_bytes =
        summary.descriptor_gpu_resident_h2d_bytes + summary.descriptor_gpu_staged_h2d_bytes;
    if descriptor_bytes == 0 {
        record_h2d(ledger, summary.h2d_bytes, "hf_cuda_sequence_bootstrap_h2d");
        return;
    }
    record_h2d(
        ledger,
        summary.descriptor_gpu_resident_h2d_bytes,
        "hf_cuda_sequence_descriptor_resident_h2d",
    );
    record_h2d(
        ledger,
        summary.descriptor_gpu_staged_h2d_bytes,
        "hf_cuda_sequence_descriptor_staged_h2d",
    );
    record_h2d(
        ledger,
        summary.h2d_bytes.saturating_sub(descriptor_bytes),
        "hf_cuda_sequence_metadata_h2d",
    );
}

fn record_decision(ledger: &mut TokenLedger, summary: &CudaHfDecodeSequenceSummary, _step: usize) {
    let visible = visible_ns(summary);
    let device = device_active_ns(summary);
    let measured = summary.device_elapsed_ns > 0;
    ledger.record_execution_decision(ExecutionDecision {
        operation: "hf_cuda_device_token_sequence",
        executor_selected: ExecutionOwner::Gpu(DeviceOrdinal(0)),
        candidate_costs: vec![
            if measured {
                CandidateCost::measured("cuda-device-token-sequence", device)
            } else {
                CandidateCost::estimated("cuda-device-token-sequence", visible)
            },
            CandidateCost::estimated("host-per-token-chain", visible.saturating_mul(2)),
        ],
        reason: "loaded HF decode sequence keeps next-token causality on device",
        predicted_visible_ns: visible,
        actual_visible_ns: measured.then_some(device),
        metric_source: device_metric_source(summary),
    });
}

fn record_final_sync(ledger: &mut TokenLedger, summary: &CudaHfDecodeSequenceSummary) {
    ledger.record_sync(
        SyncClass::SoftVisibilitySync,
        None,
        Some(MemoryTier::Vram),
        Some(MemoryTier::PinnedDram),
        summary.d2h_bytes as usize,
        summary.sync_calls.max(1),
        MetricSource::EstimatedModel,
        "hf_cuda_sequence_final_token_visibility",
    );
}

fn record_device_span(ledger: &mut TokenLedger, active_ns: u64, source: MetricSource) {
    ledger
        .record_device_span(DeviceTimelineSpan::new(
            DeviceOrdinal(0),
            0,
            active_ns,
            source,
            "hf_cuda_sequence_device_timeline",
        ))
        .expect("HF CUDA sequence ledger records valid device spans");
}

fn record_copy(ledger: &mut TokenLedger, h2d: bool, bytes: u64, label: &'static str) {
    if bytes == 0 {
        return;
    }
    let (from, to) = if h2d {
        (MemoryTier::PinnedDram, MemoryTier::Vram)
    } else {
        (MemoryTier::Vram, MemoryTier::PinnedDram)
    };
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Copy,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: None,
        from_tier: Some(from),
        to_tier: Some(to),
        bytes: bytes as usize,
        latency_ns: bytes.max(1),
        label,
    });
}

fn record_h2d(ledger: &mut TokenLedger, bytes: u64, label: &'static str) {
    record_copy(ledger, true, bytes, label);
}

fn record_event(
    ledger: &mut TokenLedger,
    kind: LedgerEventKind,
    latency_ns: u64,
    metric_source: MetricSource,
    label: &'static str,
) {
    ledger.record(LedgerEvent {
        kind,
        sync_class: None,
        metric_source,
        block_id: None,
        from_tier: Some(MemoryTier::Vram),
        to_tier: Some(MemoryTier::Vram),
        bytes: 0,
        latency_ns,
        label,
    });
}

fn visible_ns(summary: &CudaHfDecodeSequenceSummary) -> u64 {
    let token_count = summary.tokens.len().max(1) as u64;
    let copy = (summary.h2d_bytes + summary.d2h_bytes) / token_count;
    (summary.resident_weight_bytes / token_count + copy).max(1)
}

fn device_active_ns(summary: &CudaHfDecodeSequenceSummary) -> u64 {
    if summary.device_elapsed_ns == 0 {
        return visible_ns(summary);
    }
    let token_count = summary.tokens.len().max(1) as u64;
    (summary.device_elapsed_ns / token_count).max(1)
}

fn device_metric_source(summary: &CudaHfDecodeSequenceSummary) -> MetricSource {
    if summary.device_elapsed_ns == 0 {
        MetricSource::EstimatedModel
    } else {
        MetricSource::GpuEvent
    }
}

fn graph_replay_ns(summary: &CudaHfDecodeSequenceSummary) -> u64 {
    let replay_count = summary.graph_replays.max(1);
    ((summary.graph_launches.max(1) + summary.graph_nodes.max(1)) / replay_count).max(1)
}
