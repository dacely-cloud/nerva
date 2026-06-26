use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::transport::tcp_control::config::{TcpControlProbeConfig, validate_tcp_control_config};
use crate::transport::tcp_control::summary::{TcpControlStatus, TcpControlSummary};

const ACK: &[u8] = b"control_ack_v1";

pub fn run_tcp_control_probe(config: TcpControlProbeConfig) -> Result<TcpControlSummary> {
    validate_tcp_control_config(config)?;

    let payload = control_payload(config.control_bytes);
    let listener = TcpListener::bind("127.0.0.1:0").map_err(tcp_error)?;
    let address = listener.local_addr().map_err(tcp_error)?;

    let start = Instant::now();
    let mut client = TcpStream::connect(address).map_err(tcp_error)?;
    let (mut server, _) = listener.accept().map_err(tcp_error)?;
    configure_stream(&client)?;
    configure_stream(&server)?;

    client.write_all(&payload).map_err(tcp_error)?;
    let mut inbound = vec![0u8; payload.len()];
    server.read_exact(&mut inbound).map_err(tcp_error)?;
    if inbound != payload {
        return Err(NervaError::InvalidArgument {
            reason: "TCP control loopback payload mismatch".to_string(),
        });
    }

    server.write_all(ACK).map_err(tcp_error)?;
    server.flush().map_err(tcp_error)?;
    let mut ack = vec![0u8; ACK.len()];
    client.read_exact(&mut ack).map_err(tcp_error)?;
    if ack != ACK {
        return Err(NervaError::InvalidArgument {
            reason: "TCP control loopback ACK mismatch".to_string(),
        });
    }
    let latency_ns = elapsed_ns(start);
    let total_wire_bytes = payload.len().saturating_add(ack.len());

    let mut ledger = TokenLedger::new(0);
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Transport,
        sync_class: None,
        metric_source: MetricSource::RuntimeTimestamp,
        block_id: None,
        from_tier: Some(MemoryTier::Dram),
        to_tier: Some(MemoryTier::Dram),
        bytes: total_wire_bytes,
        latency_ns,
        label: "tcp_control_loopback",
    });
    ledger.require_zero_hot_path_allocations()?;

    Ok(TcpControlSummary {
        status: TcpControlStatus::Ok,
        backend: "tcp_control_only",
        protocol_version: config.protocol_version,
        request_id: config.request_id,
        sequence_id: config.sequence_id,
        control_bytes_sent: payload.len(),
        control_bytes_received: ack.len(),
        tensor_payload_bytes: 0,
        total_wire_bytes,
        connection_count: 1,
        control_messages: 2,
        completion_latency_ns: latency_ns,
        effective_control_bandwidth_bps: bandwidth_bps(total_wire_bytes, latency_ns),
        runtime_timestamp_events: ledger.event_count_for_source(MetricSource::RuntimeTimestamp),
        transport_events: ledger.event_count(LedgerEventKind::Transport),
        control_plane_only: true,
        debug_only: true,
        production_tensor_data_plane: false,
        pageable_copies: 0,
        per_token_registrations: 0,
        hot_path_allocations: ledger.hot_path_allocations,
    })
}

fn configure_stream(stream: &TcpStream) -> Result<()> {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(tcp_error)?;
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .map_err(tcp_error)
}

fn control_payload(bytes: usize) -> Vec<u8> {
    (0..bytes)
        .map(|index| (index as u8).wrapping_mul(31))
        .collect()
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn bandwidth_bps(bytes: usize, elapsed_ns: u64) -> u64 {
    if elapsed_ns == 0 {
        0
    } else {
        ((bytes as u128).saturating_mul(1_000_000_000) / elapsed_ns as u128).min(u64::MAX as u128)
            as u64
    }
}

fn tcp_error(err: std::io::Error) -> NervaError {
    NervaError::BackendUnavailable {
        backend: "tcp_control_only",
        reason: err.to_string(),
    }
}
