#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KernelUdpProbeConfig {
    pub protocol_version: u16,
    pub request_id: u64,
    pub sequence_id: u64,
    pub block_id: u64,
    pub block_version: u64,
    pub payload_bytes: usize,
    pub chunk_payload_bytes: usize,
}

impl KernelUdpProbeConfig {
    pub const fn reference_decode_activation() -> Self {
        Self {
            protocol_version: 1,
            request_id: 1,
            sequence_id: 1,
            block_id: 32_768,
            block_version: 7,
            payload_bytes: 32 * 1024,
            chunk_payload_bytes: 4 * 1024,
        }
    }

    pub fn chunk_count(self) -> usize {
        self.payload_bytes.div_ceil(self.chunk_payload_bytes)
    }
}
