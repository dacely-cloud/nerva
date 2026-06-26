use nerva_core::types::error::{NervaError, Result};

use crate::transport::kernel_udp::config::KernelUdpProbeConfig;
use crate::transport::kernel_udp::packet::{HEADER_BYTES, KernelUdpPacketHeader};

pub(super) fn validate_config(config: KernelUdpProbeConfig) -> Result<()> {
    if config.protocol_version == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "kernel UDP protocol version must be nonzero".to_string(),
        });
    }
    if config.payload_bytes == 0 || config.chunk_payload_bytes == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "kernel UDP payload and chunk sizes must be nonzero".to_string(),
        });
    }
    if HEADER_BYTES.saturating_add(config.chunk_payload_bytes) > 60 * 1024 {
        return Err(NervaError::InvalidArgument {
            reason: "kernel UDP chunk size exceeds safe loopback datagram size".to_string(),
        });
    }
    Ok(())
}

pub(super) fn validate_header(
    config: KernelUdpProbeConfig,
    chunk_id: usize,
    chunk_count: usize,
    offset: usize,
    length: usize,
    header: KernelUdpPacketHeader,
) -> Result<()> {
    let expected = (
        config.protocol_version,
        config.request_id,
        config.sequence_id,
        config.block_id,
        config.block_version,
        chunk_id as u32,
        chunk_count as u32,
        offset as u32,
        length as u32,
    );
    let observed = (
        header.protocol_version,
        header.request_id,
        header.sequence_id,
        header.block_id,
        header.block_version,
        header.chunk_id,
        header.chunk_count,
        header.offset,
        header.length,
    );
    if observed == expected {
        Ok(())
    } else {
        Err(NervaError::InvalidArgument {
            reason: "kernel UDP packet identity mismatch".to_string(),
        })
    }
}
