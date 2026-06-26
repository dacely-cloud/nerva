use crate::capabilities::snapshot::CapabilityState;
use crate::transport::dpdk_udp::config::{DpdkUdpProbeConfig, validate_dpdk_udp_config};
use nerva_core::types::error::{NervaError, Result};

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

    let (selected_path, capability_result, pinned_host_required, direct_gpu_memory_claimed) =
        if dpdk_udp_gpu == CapabilityState::SupportedAndVerified {
            (
                DpdkUdpMemoryPath::GpuBuffer,
                CapabilityState::SupportedAndVerified,
                false,
                true,
            )
        } else if matches!(
            dpdk_udp_pinned_host,
            CapabilityState::SupportedAndVerified | CapabilityState::SupportedUnverified
        ) {
            (
                DpdkUdpMemoryPath::PinnedHostBuffer,
                CapabilityState::DegradedToPinnedHost,
                true,
                false,
            )
        } else {
            return Err(NervaError::BackendUnavailable {
                backend: "dpdk_udp",
                reason: "no verified GPU-buffer path and pinned-host DPDK UDP is unavailable"
                    .to_string(),
            });
        };

    let chunks = plan_chunks(config, chunk_count)?;
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
    let retransmit_payload_bytes = chunks
        .iter()
        .filter(|chunk| chunk.needs_nack)
        .map(|chunk| chunk.bytes)
        .sum::<usize>();
    let total_payload_bytes = config
        .payload_bytes
        .checked_add(retransmit_payload_bytes)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "DPDK UDP payload byte count overflowed".to_string(),
        })?;
    let total_wire_bytes = total_payload_bytes
        .checked_add(protocol_header_bytes)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "DPDK UDP wire byte count overflowed".to_string(),
        })?;
    let credit_windows = div_ceil_u32(chunk_count, config.credit_window_chunks);
    let credit_stalls = credit_windows.saturating_sub(1);
    let receiver_bitmap_words = div_ceil_u32(config.receiver_bitmap_chunks, 64);

    Ok(DpdkUdpProtocolPlan {
        config,
        selected_path,
        capability_result,
        pinned_host_required,
        direct_gpu_memory_claimed,
        chunks,
        chunk_count,
        total_payload_bytes,
        protocol_header_bytes,
        total_wire_bytes,
        preposted_receives: chunk_count,
        credit_windows,
        credit_stalls,
        sender_retention_chunks: config.sender_retention_chunks,
        receiver_bitmap_words,
        nack_ranges,
        selective_retransmits,
        ack_packets: 0,
        mbufs_preallocated: total_packets.saturating_mul(2),
        rings_preallocated: 2,
    })
}

fn plan_chunks(config: DpdkUdpProbeConfig, chunk_count: u32) -> Result<Vec<DpdkUdpChunkPlan>> {
    let mut chunks = Vec::with_capacity(chunk_count as usize);
    for chunk_id in 0..chunk_count {
        let offset = (chunk_id as usize)
            .checked_mul(config.chunk_payload_bytes)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "DPDK UDP chunk offset overflowed".to_string(),
            })?;
        let remaining = config.payload_bytes.saturating_sub(offset);
        let bytes = remaining.min(config.chunk_payload_bytes);
        let needs_nack = config.packet_loss_period > 0
            && (chunk_id + 1) % config.packet_loss_period == 0
            && chunk_id + 1 < chunk_count;
        chunks.push(DpdkUdpChunkPlan {
            chunk_id,
            offset,
            bytes,
            retained_by_sender: chunk_id < config.sender_retention_chunks,
            receiver_bitmap_bit: chunk_id,
            needs_nack,
            retransmit_attempts: u32::from(needs_nack),
        });
    }
    Ok(chunks)
}

fn div_ceil_usize(value: usize, divisor: usize) -> Result<u32> {
    let count = value
        .checked_add(divisor - 1)
        .and_then(|sum| sum.checked_div(divisor))
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "DPDK UDP chunk count overflowed".to_string(),
        })?;
    u32::try_from(count).map_err(|_| NervaError::InvalidArgument {
        reason: "DPDK UDP chunk count exceeds u32".to_string(),
    })
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.saturating_add(divisor.saturating_sub(1)) / divisor
}
