use crate::transport::dpdk_udp::config::DpdkUdpProbeConfig;
use crate::transport::dpdk_udp::protocol::DpdkUdpChunkPlan;
use nerva_core::types::error::{NervaError, Result};

pub(super) struct DpdkUdpWireTotals {
    pub total_packets: u32,
    pub total_payload_bytes: usize,
    pub protocol_header_bytes: usize,
    pub total_wire_bytes: usize,
    pub nack_ranges: u32,
    pub selective_retransmits: u32,
}

pub(super) fn compute_wire_totals(
    config: DpdkUdpProbeConfig,
    chunk_count: u32,
    chunks: &[DpdkUdpChunkPlan],
) -> Result<DpdkUdpWireTotals> {
    let nack_ranges = chunks.iter().filter(|chunk| chunk.needs_nack).count() as u32;
    let selective_retransmits = chunks
        .iter()
        .map(|chunk| chunk.retransmit_attempts)
        .sum::<u32>();
    let total_packets = chunk_count
        .checked_add(selective_retransmits)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "DPDK UDP packet count overflowed".to_string(),
        })?;
    let protocol_header_bytes = config
        .protocol_header_bytes
        .checked_mul(total_packets as usize)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "DPDK UDP protocol header byte count overflowed".to_string(),
        })?;
    let total_payload_bytes = config
        .payload_bytes
        .checked_add(retransmit_payload_bytes(chunks))
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "DPDK UDP payload byte count overflowed".to_string(),
        })?;
    let total_wire_bytes = total_payload_bytes
        .checked_add(protocol_header_bytes)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "DPDK UDP wire byte count overflowed".to_string(),
        })?;

    Ok(DpdkUdpWireTotals {
        total_packets,
        total_payload_bytes,
        protocol_header_bytes,
        total_wire_bytes,
        nack_ranges,
        selective_retransmits,
    })
}

fn retransmit_payload_bytes(chunks: &[DpdkUdpChunkPlan]) -> usize {
    chunks
        .iter()
        .filter(|chunk| chunk.needs_nack)
        .map(|chunk| chunk.bytes)
        .sum()
}
