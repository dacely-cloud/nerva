use crate::capabilities::snapshot::CapabilityState;
use crate::transport::dpdk_udp::config::DpdkUdpProbeConfig;
use crate::transport::dpdk_udp::protocol::{DpdkUdpMemoryPath, plan_dpdk_udp_protocol};
use crate::transport::dpdk_udp::summary::{DpdkUdpProtocolStatus, DpdkUdpProtocolSummary};
use nerva_core::types::error::Result;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::fallback::{FallbackClass, FallbackDecision};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

pub fn run_dpdk_udp_protocol_probe(
    config: DpdkUdpProbeConfig,
    dpdk_udp_gpu: CapabilityState,
    dpdk_udp_pinned_host: CapabilityState,
) -> Result<DpdkUdpProtocolSummary> {
    let plan = plan_dpdk_udp_protocol(config, dpdk_udp_gpu, dpdk_udp_pinned_host)?;
    let mut ledger = TokenLedger::new(0);

    if plan.selected_path == DpdkUdpMemoryPath::PinnedHostBuffer {
        ledger.record_fallback_decision(FallbackDecision {
            label: "dpdk_udp_pinned_host_fallback",
            class: FallbackClass::CapabilityDegraded,
            requested: "dpdk_udp_gpu",
            selected: "dpdk_udp_pinned_host",
            reason: "DPDK GPU-buffer path is not verified; using preallocated pinned-host mbufs",
            visible_ns: Some(plan.total_wire_bytes as u64),
            metric_source: MetricSource::EstimatedModel,
        });
    }

    for chunk in &plan.chunks {
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Transport,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: None,
            from_tier: Some(MemoryTier::PinnedDram),
            to_tier: Some(MemoryTier::PinnedDram),
            bytes: chunk.bytes.saturating_add(config.protocol_header_bytes),
            latency_ns: chunk.bytes as u64,
            label: "dpdk_udp_chunk_tx",
        });
        if chunk.needs_nack {
            ledger.record(LedgerEvent {
                kind: LedgerEventKind::Transport,
                sync_class: None,
                metric_source: MetricSource::EstimatedModel,
                block_id: None,
                from_tier: Some(MemoryTier::PinnedDram),
                to_tier: Some(MemoryTier::PinnedDram),
                bytes: chunk.bytes.saturating_add(config.protocol_header_bytes),
                latency_ns: chunk.bytes as u64,
                label: "dpdk_udp_selective_retransmit",
            });
        }
    }
    for _ in 0..plan.credit_stalls {
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Transport,
            sync_class: None,
            metric_source: MetricSource::EstimatedModel,
            block_id: None,
            from_tier: Some(MemoryTier::PinnedDram),
            to_tier: Some(MemoryTier::PinnedDram),
            bytes: 0,
            latency_ns: config.credit_stall_ns_per_window,
            label: "dpdk_udp_credit_window_stall",
        });
    }

    ledger.record_sync(
        SyncClass::PhaseHandoff,
        None,
        Some(MemoryTier::PinnedDram),
        Some(MemoryTier::PinnedDram),
        0,
        1,
        MetricSource::RuntimeTimestamp,
        "dpdk_udp_range_completion_handoff",
    );
    ledger.require_zero_hot_path_allocations()?;
    ledger.require_classified_syncs()?;

    let summary = DpdkUdpProtocolSummary {
        status: DpdkUdpProtocolStatus::Ok,
        protocol_version: config.protocol_version,
        request_id: config.request_id,
        sequence_id: config.sequence_id,
        block_id: config.block_id,
        block_version: config.block_version,
        mode: config.mode,
        selected_path: plan.selected_path,
        capability_result: plan.capability_result,
        payload_bytes: config.payload_bytes,
        chunk_payload_bytes: config.chunk_payload_bytes,
        chunks: plan.chunk_count,
        protocol_header_bytes: plan.protocol_header_bytes,
        total_wire_bytes: plan.total_wire_bytes,
        preposted_receives: plan.preposted_receives,
        credit_window_chunks: config.credit_window_chunks,
        credit_windows: plan.credit_windows,
        credit_stalls: plan.credit_stalls,
        credit_stall_ns: plan.credit_stall_ns,
        sender_retention_chunks: plan.sender_retention_chunks,
        receiver_bitmap_words: plan.receiver_bitmap_words,
        nack_ranges: plan.nack_ranges,
        selective_retransmits: plan.selective_retransmits,
        ack_packets: plan.ack_packets,
        mbufs_preallocated: plan.mbufs_preallocated,
        rings_preallocated: plan.rings_preallocated,
        direct_gpu_memory_claimed: plan.direct_gpu_memory_claimed,
        pinned_host_required: plan.pinned_host_required,
        fallback_decisions: ledger.fallback_count(),
        transport_events: ledger.event_count(LedgerEventKind::Transport),
        phase_handoff_syncs: ledger.sync_count_for(SyncClass::PhaseHandoff),
        pageable_copies: 0,
        per_token_registrations: 0,
        hot_path_allocations: ledger.hot_path_allocations,
        error: None,
    };
    Ok(summary)
}
