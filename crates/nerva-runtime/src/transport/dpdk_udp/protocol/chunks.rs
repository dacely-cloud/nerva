use crate::transport::dpdk_udp::config::DpdkUdpProbeConfig;
use crate::transport::dpdk_udp::protocol::DpdkUdpChunkPlan;
use nerva_core::types::error::{NervaError, Result};

pub(super) fn plan_chunks(
    config: DpdkUdpProbeConfig,
    chunk_count: u32,
) -> Result<Vec<DpdkUdpChunkPlan>> {
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
