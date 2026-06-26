use crate::capabilities::snapshot::CapabilityState;
use crate::transport::dpdk_udp::config::{DpdkUdpProbeConfig, validate_dpdk_udp_config};
use crate::transport::dpdk_udp::math::{div_ceil_u32, div_ceil_usize};
use crate::transport::dpdk_udp::protocol::chunks::plan_chunks;
use crate::transport::dpdk_udp::protocol::path::select_memory_path;
use crate::transport::dpdk_udp::protocol::wire::compute_wire_totals;
use nerva_core::types::error::{NervaError, Result};

mod chunks;
mod path;
mod wire;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DpdkUdpMemoryPath {
    GpuBuffer,
    PinnedHostBuffer,
}

impl DpdkUdpMemoryPath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GpuBuffer => "dpdk_udp_gpu",
            Self::PinnedHostBuffer => "dpdk_udp_pinned_host",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DpdkUdpChunkPlan {
    pub chunk_id: u32,
    pub offset: usize,
    pub bytes: usize,
    pub retained_by_sender: bool,
    pub receiver_bitmap_bit: u32,
    pub needs_nack: bool,
    pub retransmit_attempts: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DpdkUdpProtocolPlan {
    pub config: DpdkUdpProbeConfig,
    pub selected_path: DpdkUdpMemoryPath,
    pub capability_result: CapabilityState,
    pub pinned_host_required: bool,
    pub direct_gpu_memory_claimed: bool,
    pub chunks: Vec<DpdkUdpChunkPlan>,
    pub chunk_count: u32,
    pub total_payload_bytes: usize,
    pub protocol_header_bytes: usize,
    pub total_wire_bytes: usize,
    pub preposted_receives: u32,
    pub credit_windows: u32,
    pub credit_stalls: u32,
    pub credit_stall_ns: u64,
    pub sender_retention_chunks: u32,
    pub receiver_bitmap_words: u32,
    pub nack_ranges: u32,
    pub selective_retransmits: u32,
    pub ack_packets: u32,
    pub mbufs_preallocated: u32,
    pub rings_preallocated: u32,
}

pub fn plan_dpdk_udp_protocol(
    config: DpdkUdpProbeConfig,
    dpdk_udp_gpu: CapabilityState,
    dpdk_udp_pinned_host: CapabilityState,
) -> Result<DpdkUdpProtocolPlan> {
    validate_dpdk_udp_config(config)?;
    let chunk_count = div_ceil_usize(config.payload_bytes, config.chunk_payload_bytes)?;
    if chunk_count > config.receiver_bitmap_chunks {
        return Err(NervaError::InvalidArgument {
            reason: "DPDK UDP receiver bitmap cannot represent all chunks".to_string(),
        });
    }
    if chunk_count > config.sender_retention_chunks {
        return Err(NervaError::InvalidArgument {
            reason: "DPDK UDP sender retention cannot cover all chunks".to_string(),
        });
    }

    let selected = select_memory_path(dpdk_udp_gpu, dpdk_udp_pinned_host)?;

    let chunks = plan_chunks(config, chunk_count)?;
    let wire = compute_wire_totals(config, chunk_count, &chunks)?;
    let credit_windows = div_ceil_u32(chunk_count, config.credit_window_chunks);
    let credit_stalls = credit_windows.saturating_sub(1);
    let credit_stall_ns = u64::from(credit_stalls)
        .checked_mul(config.credit_stall_ns_per_window)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "DPDK UDP credit stall cost overflowed".to_string(),
        })?;
    let receiver_bitmap_words = div_ceil_u32(config.receiver_bitmap_chunks, 64);

    Ok(DpdkUdpProtocolPlan {
        config,
        selected_path: selected.selected_path,
        capability_result: selected.capability_result,
        pinned_host_required: selected.pinned_host_required,
        direct_gpu_memory_claimed: selected.direct_gpu_memory_claimed,
        chunks,
        chunk_count,
        total_payload_bytes: wire.total_payload_bytes,
        protocol_header_bytes: wire.protocol_header_bytes,
        total_wire_bytes: wire.total_wire_bytes,
        preposted_receives: chunk_count,
        credit_windows,
        credit_stalls,
        credit_stall_ns,
        sender_retention_chunks: config.sender_retention_chunks,
        receiver_bitmap_words,
        nack_ranges: wire.nack_ranges,
        selective_retransmits: wire.selective_retransmits,
        ack_packets: 0,
        mbufs_preallocated: wire.total_packets.saturating_mul(2),
        rings_preallocated: 2,
    })
}
