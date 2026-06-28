use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;

use crate::engine::hf_cuda_decode::file_backed::session_stream_types::{
    HfCudaDeviceSessionStreamRecord, HfCudaHostOutputQueueSummary,
};

pub(super) struct BoundedHostOutputQueue {
    slots: Vec<Option<HfCudaDeviceSessionStreamRecord>>,
    next_slot: usize,
    len: usize,
    high_watermark: usize,
    pushes: u64,
    drains: u64,
    overflows: u64,
}

impl BoundedHostOutputQueue {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            slots: vec![None; capacity],
            next_slot: 0,
            len: 0,
            high_watermark: 0,
            pushes: 0,
            drains: 0,
            overflows: 0,
        }
    }

    pub(super) fn push(
        &mut self,
        token: TokenId,
        chunk_index: usize,
        chunk_offset: usize,
    ) -> Result<HfCudaDeviceSessionStreamRecord> {
        if self.len == self.slots.len() {
            self.overflows += 1;
            return Err(NervaError::InvalidArgument {
                reason: "HF CUDA session stream host output queue overflow".to_string(),
            });
        }
        let slot = self.next_slot;
        let record = HfCudaDeviceSessionStreamRecord {
            token_index: self.pushes,
            token,
            chunk_index,
            chunk_offset,
            queue_slot: slot,
            host_visible_order: self.pushes,
            device_authoritative: true,
            host_causality_edge: false,
        };
        self.slots[slot] = Some(record.clone());
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        self.len += 1;
        self.pushes += 1;
        self.high_watermark = self.high_watermark.max(self.len);
        Ok(record)
    }

    pub(super) fn drain_all(&mut self) {
        self.drains += self.len as u64;
        self.slots.iter_mut().for_each(|slot| *slot = None);
        self.len = 0;
    }

    pub(super) fn summary(&self) -> HfCudaHostOutputQueueSummary {
        HfCudaHostOutputQueueSummary {
            capacity: self.slots.len(),
            pushes: self.pushes,
            drains: self.drains,
            high_watermark: self.high_watermark,
            overflow_rejections: self.overflows,
            host_causality_edges: 0,
        }
    }
}
