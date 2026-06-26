use std::time::{Duration, Instant};

use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::runtime::Runtime;
use crate::transport::kernel_udp::config::KernelUdpProbeConfig;
use crate::transport::kernel_udp::io::{bind_loopback_socket, elapsed_ns, io_error};
use crate::transport::kernel_udp::packet::{HEADER_BYTES, decode_packet, encode_packet, payload};
use crate::transport::kernel_udp::payload_data::deterministic_payload;
use crate::transport::kernel_udp::stats::{bandwidth_bps, latency_stats};
use crate::transport::kernel_udp::summary::{KernelUdpBaselineStatus, KernelUdpBaselineSummary};
use crate::transport::kernel_udp::validate::{validate_config, validate_header};

impl Runtime {
    pub fn run_kernel_udp_baseline_probe(
        &self,
        config: KernelUdpProbeConfig,
    ) -> Result<KernelUdpBaselineSummary> {
        let _ = self.config();
        run_kernel_udp_baseline_probe(config)
    }
}

pub fn run_kernel_udp_baseline_probe(
    config: KernelUdpProbeConfig,
) -> Result<KernelUdpBaselineSummary> {
    validate_config(config)?;

    let receiver = bind_loopback_socket()?;
    let sender = bind_loopback_socket()?;
    receiver
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(io_error)?;
    sender
        .set_write_timeout(Some(Duration::from_secs(2)))
        .map_err(io_error)?;
    let receiver_addr = receiver.local_addr().map_err(io_error)?;
    let sender_addr = sender.local_addr().map_err(io_error)?;

    let chunk_count = config.chunk_count();
    let source_payload = deterministic_payload(config.payload_bytes);
    let mut received_payload = vec![0u8; config.payload_bytes];
    let mut send_packet = vec![0u8; HEADER_BYTES + config.chunk_payload_bytes];
    let mut recv_packet = vec![0u8; HEADER_BYTES + config.chunk_payload_bytes];
    let mut latencies = Vec::with_capacity(chunk_count);
    let mut ledger = TokenLedger::new(0);
    ledger.events.reserve_exact(chunk_count);

    let mut packets_sent = 0u64;
    let mut packets_received = 0u64;
    let mut validated_packets = 0u64;
    let mut bytes_received = 0usize;
    let mut total_wire_bytes = 0usize;

    for chunk_id in 0..chunk_count {
        let offset = chunk_id.saturating_mul(config.chunk_payload_bytes);
        let end = offset
            .saturating_add(config.chunk_payload_bytes)
            .min(config.payload_bytes);
        let chunk = &source_payload[offset..end];
        let packet_len = encode_packet(
            &mut send_packet,
            config,
            chunk_id,
            chunk_count,
            offset,
            chunk,
        )?;

        let start = Instant::now();
        let sent = sender
            .send_to(&send_packet[..packet_len], receiver_addr)
            .map_err(io_error)?;
        packets_sent = packets_sent.saturating_add(1);
        if sent != packet_len {
            return Err(NervaError::InvalidArgument {
                reason: "kernel UDP sent a partial datagram".to_string(),
            });
        }

        let (received, source_addr) = receiver.recv_from(&mut recv_packet).map_err(io_error)?;
        let elapsed_ns = elapsed_ns(start);
        if source_addr != sender_addr {
            return Err(NervaError::InvalidArgument {
                reason: "kernel UDP received a datagram from an unexpected sender".to_string(),
            });
        }
        packets_received = packets_received.saturating_add(1);
        total_wire_bytes = total_wire_bytes.saturating_add(received);

        match decode_packet(&recv_packet[..received]) {
            Ok(header) => {
                validate_header(config, chunk_id, chunk_count, offset, chunk.len(), header)?;
                let received_chunk = payload(&recv_packet[..received]);
                let destination = &mut received_payload[offset..end];
                destination.copy_from_slice(received_chunk);
                bytes_received = bytes_received.saturating_add(received_chunk.len());
                validated_packets = validated_packets.saturating_add(1);
            }
            Err(err) => {
                return Err(err);
            }
        }

        latencies.push(elapsed_ns);
        ledger.record(LedgerEvent {
            kind: LedgerEventKind::Transport,
            sync_class: None,
            metric_source: MetricSource::RuntimeTimestamp,
            block_id: None,
            from_tier: Some(MemoryTier::Dram),
            to_tier: Some(MemoryTier::Dram),
            bytes: received,
            latency_ns: elapsed_ns,
            label: "kernel_udp_loopback_chunk",
        });
    }

    if received_payload != source_payload {
        return Err(NervaError::InvalidArgument {
            reason: "kernel UDP reconstructed payload mismatch".to_string(),
        });
    }

    ledger.require_zero_hot_path_allocations()?;
    let stats = latency_stats(&latencies);

    Ok(KernelUdpBaselineSummary {
        status: KernelUdpBaselineStatus::Ok,
        backend: "kernel_udp_test",
        protocol_version: config.protocol_version,
        request_id: config.request_id,
        sequence_id: config.sequence_id,
        block_id: config.block_id,
        block_version: config.block_version,
        payload_bytes: config.payload_bytes,
        chunk_payload_bytes: config.chunk_payload_bytes,
        chunks: chunk_count as u64,
        protocol_header_bytes: HEADER_BYTES,
        total_wire_bytes,
        packets_sent,
        packets_received,
        validated_packets,
        bytes_received,
        p50_completion_latency_ns: stats.p50_ns,
        p95_completion_latency_ns: stats.p95_ns,
        p99_completion_latency_ns: stats.p99_ns,
        total_completion_latency_ns: stats.total_ns,
        effective_payload_bandwidth_bps: bandwidth_bps(config.payload_bytes, stats.total_ns),
        runtime_timestamp_events: ledger.event_count_for_source(MetricSource::RuntimeTimestamp),
        transport_events: ledger.event_count(LedgerEventKind::Transport),
        packet_loss: packets_sent.saturating_sub(packets_received),
        checksum_failures: 0,
        baseline_only: true,
        production_tensor_data_plane: false,
        pageable_copies: 0,
        per_token_registrations: 0,
        hot_path_allocations: ledger.hot_path_allocations,
    })
}
