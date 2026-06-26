use nerva_core::types::error::{NervaError, Result};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TcpControlProbeConfig {
    pub protocol_version: u16,
    pub request_id: u64,
    pub sequence_id: u64,
    pub control_bytes: usize,
    pub max_control_bytes: usize,
}

impl TcpControlProbeConfig {
    pub const fn reference_handshake() -> Self {
        Self {
            protocol_version: 1,
            request_id: 1,
            sequence_id: 1,
            control_bytes: 64,
            max_control_bytes: 1024,
        }
    }
}

pub(crate) fn validate_tcp_control_config(config: TcpControlProbeConfig) -> Result<()> {
    if config.protocol_version == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "TCP control protocol version must be non-zero".to_string(),
        });
    }
    if config.request_id == 0 || config.sequence_id == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "TCP control identifiers must be non-zero".to_string(),
        });
    }
    if config.control_bytes == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "TCP control payload must be non-empty".to_string(),
        });
    }
    if config.control_bytes > config.max_control_bytes {
        return Err(NervaError::InvalidArgument {
            reason: "TCP control payload exceeds configured control-plane limit".to_string(),
        });
    }
    Ok(())
}
