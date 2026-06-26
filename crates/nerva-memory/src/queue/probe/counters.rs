use crate::queue::types::SharedQueueRejectionKind;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct SharedQueueCounters {
    pub(crate) descriptors_posted: u64,
    pub(crate) descriptors_completed: u64,
    pub(crate) queue_full_rejections: u64,
    pub(crate) wrong_producer_rejections: u64,
    pub(crate) wrong_consumer_rejections: u64,
    pub(crate) bulk_payload_rejections: u64,
    pub(crate) payload_bytes_in_queue: u64,
    pub(crate) referenced_block_bytes: u64,
}

impl SharedQueueCounters {
    pub(crate) const fn new() -> Self {
        Self {
            descriptors_posted: 0,
            descriptors_completed: 0,
            queue_full_rejections: 0,
            wrong_producer_rejections: 0,
            wrong_consumer_rejections: 0,
            bulk_payload_rejections: 0,
            payload_bytes_in_queue: 0,
            referenced_block_bytes: 0,
        }
    }

    pub(crate) fn record_rejection(&mut self, kind: SharedQueueRejectionKind) {
        match kind {
            SharedQueueRejectionKind::QueueFull => self.queue_full_rejections += 1,
            SharedQueueRejectionKind::WrongProducer => self.wrong_producer_rejections += 1,
            SharedQueueRejectionKind::WrongConsumer => self.wrong_consumer_rejections += 1,
            SharedQueueRejectionKind::BulkPayloadInDescriptor => self.bulk_payload_rejections += 1,
            SharedQueueRejectionKind::QueueEmpty | SharedQueueRejectionKind::InvalidQueueBlocks => {
            }
        }
    }
}
