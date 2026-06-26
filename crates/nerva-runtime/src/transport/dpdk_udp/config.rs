use crate::transport::path::types::TransferMode;
use nerva_core::types::error::{NervaError, Result};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DpdkUdpProbeConfig {
    pub protocol_version: u32,
    pub request_id: u64,
    pub sequence_id: u64,
    pub block_id: u64,
    pub block_version: u64,
    pub payload_bytes: usize,
    pub chunk_payload_bytes: usize,
    pub protocol_header_bytes: usize,
    pub mode: TransferMode,
    pub credit_window_chunks: u32,
    pub sender_retention_chunks: u32,
    pub receiver_bitmap_chunks: u32,
    pub packet_loss_period: u32,
}

impl DpdkUdpProbeConfig {
    pub const fn reference_decode_activation() -> Self {
        Self {
            protocol_version: 1,
            request_id: 1,
            sequence_id: 1,
            block_id: 9001,
            block_version: 7,
            payload_bytes: 32 * 1024,
            chunk_payload_bytes: 4 * 1024,
            protocol_header_bytes: 64,
            mode: TransferMode::Decode,
            credit_window_chunks: 8,
            sender_retention_chunks: 16,
            receiver_bitmap_chunks: 64,
            packet_loss_period: 5,
        }
    }

    pub const fn credit_pressure_decode_activation() -> Self {
        let mut config = Self::reference_decode_activation();
        config.request_id = 2;
        config.sequence_id = 2;
        config.credit_window_chunks = 3;
        config.packet_loss_period = 0;
        config
    }
}

pub(crate) fn validate_dpdk_udp_config(config: DpdkUdpProbeConfig) -> Result<()> {
    if config.protocol_version == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DPDK UDP protocol version must be non-zero".to_string(),
        });
    }
    if config.request_id == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DPDK UDP request id must be non-zero".to_string(),
        });
    }
    if config.sequence_id == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DPDK UDP sequence id must be non-zero".to_string(),
        });
    }
    if config.block_id == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DPDK UDP block id must be non-zero".to_string(),
        });
    }
    if config.block_version == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DPDK UDP block version must be non-zero".to_string(),
        });
    }
    if config.payload_bytes == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DPDK UDP payload bytes must be non-zero".to_string(),
        });
    }
    if config.chunk_payload_bytes == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DPDK UDP chunk payload bytes must be non-zero".to_string(),
        });
    }
    if config.protocol_header_bytes == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DPDK UDP protocol header bytes must be non-zero".to_string(),
        });
    }
    if config.credit_window_chunks == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DPDK UDP credit window must be non-zero".to_string(),
        });
    }
    if config.sender_retention_chunks == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DPDK UDP sender retention must be non-zero".to_string(),
        });
    }
    if config.receiver_bitmap_chunks == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DPDK UDP receiver bitmap capacity must be non-zero".to_string(),
        });
    }
    Ok(())
}
