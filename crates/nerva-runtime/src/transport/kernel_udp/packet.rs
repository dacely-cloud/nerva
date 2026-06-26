use nerva_core::types::error::{NervaError, Result};

use crate::transport::kernel_udp::config::KernelUdpProbeConfig;

pub(crate) const HEADER_BYTES: usize = 72;
const MAGIC: [u8; 8] = *b"NERVAUDP";

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct KernelUdpPacketHeader {
    pub protocol_version: u16,
    pub request_id: u64,
    pub sequence_id: u64,
    pub block_id: u64,
    pub block_version: u64,
    pub chunk_id: u32,
    pub chunk_count: u32,
    pub offset: u32,
    pub length: u32,
    pub checksum: u32,
}

pub(crate) fn encode_packet(
    packet: &mut [u8],
    config: KernelUdpProbeConfig,
    chunk_id: usize,
    chunk_count: usize,
    offset: usize,
    payload: &[u8],
) -> Result<usize> {
    let packet_len = HEADER_BYTES.saturating_add(payload.len());
    if packet.len() < packet_len {
        return Err(NervaError::InvalidArgument {
            reason: "kernel UDP packet buffer is too small".to_string(),
        });
    }
    if chunk_id > u32::MAX as usize || chunk_count > u32::MAX as usize {
        return Err(NervaError::InvalidArgument {
            reason: "kernel UDP chunk index exceeds protocol range".to_string(),
        });
    }
    if offset > u32::MAX as usize || payload.len() > u32::MAX as usize {
        return Err(NervaError::InvalidArgument {
            reason: "kernel UDP payload offset exceeds protocol range".to_string(),
        });
    }

    packet[..HEADER_BYTES].fill(0);
    packet[..8].copy_from_slice(&MAGIC);
    write_u16(packet, 8, config.protocol_version);
    write_u16(packet, 10, HEADER_BYTES as u16);
    write_u32(packet, 12, 0);
    write_u64(packet, 16, config.request_id);
    write_u64(packet, 24, config.sequence_id);
    write_u64(packet, 32, config.block_id);
    write_u64(packet, 40, config.block_version);
    write_u32(packet, 48, chunk_id as u32);
    write_u32(packet, 52, chunk_count as u32);
    write_u32(packet, 56, offset as u32);
    write_u32(packet, 60, payload.len() as u32);
    write_u32(packet, 64, checksum(payload));
    packet[HEADER_BYTES..packet_len].copy_from_slice(payload);
    Ok(packet_len)
}

pub(crate) fn decode_packet(packet: &[u8]) -> Result<KernelUdpPacketHeader> {
    if packet.len() < HEADER_BYTES {
        return Err(NervaError::InvalidArgument {
            reason: "kernel UDP packet is shorter than protocol header".to_string(),
        });
    }
    if packet[..8] != MAGIC[..] {
        return Err(NervaError::InvalidArgument {
            reason: "kernel UDP packet magic mismatch".to_string(),
        });
    }
    let header_bytes = read_u16(packet, 10) as usize;
    if header_bytes != HEADER_BYTES {
        return Err(NervaError::InvalidArgument {
            reason: "kernel UDP packet header length mismatch".to_string(),
        });
    }
    let length = read_u32(packet, 60);
    if HEADER_BYTES.saturating_add(length as usize) != packet.len() {
        return Err(NervaError::InvalidArgument {
            reason: "kernel UDP packet payload length mismatch".to_string(),
        });
    }
    let payload = &packet[HEADER_BYTES..];
    let observed_checksum = read_u32(packet, 64);
    if checksum(payload) != observed_checksum {
        return Err(NervaError::InvalidArgument {
            reason: "kernel UDP packet checksum mismatch".to_string(),
        });
    }
    Ok(KernelUdpPacketHeader {
        protocol_version: read_u16(packet, 8),
        request_id: read_u64(packet, 16),
        sequence_id: read_u64(packet, 24),
        block_id: read_u64(packet, 32),
        block_version: read_u64(packet, 40),
        chunk_id: read_u32(packet, 48),
        chunk_count: read_u32(packet, 52),
        offset: read_u32(packet, 56),
        length,
        checksum: observed_checksum,
    })
}

pub(crate) fn payload(packet: &[u8]) -> &[u8] {
    &packet[HEADER_BYTES..]
}

fn checksum(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811c_9dc5u32, |acc, byte| {
        acc.rotate_left(5) ^ u32::from(*byte)
    })
}

fn write_u16(packet: &mut [u8], offset: usize, value: u16) {
    packet[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(packet: &mut [u8], offset: usize, value: u32) {
    packet[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(packet: &mut [u8], offset: usize, value: u64) {
    packet[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_u16(packet: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([packet[offset], packet[offset + 1]])
}

fn read_u32(packet: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        packet[offset],
        packet[offset + 1],
        packet[offset + 2],
        packet[offset + 3],
    ])
}

fn read_u64(packet: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        packet[offset],
        packet[offset + 1],
        packet[offset + 2],
        packet[offset + 3],
        packet[offset + 4],
        packet[offset + 5],
        packet[offset + 6],
        packet[offset + 7],
    ])
}
