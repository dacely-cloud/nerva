use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::ownership::coherence::CoherencePolicy;
use nerva_core::types::ownership::owner::ExecutionOwner;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SharedWorkQueueSpec {
    pub capacity: usize,
    pub descriptor_block: ResidentBlockId,
    pub completion_block: ResidentBlockId,
    pub producer: ExecutionOwner,
    pub consumer: ExecutionOwner,
    pub coherence: CoherencePolicy,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SharedQueueDescriptor {
    pub descriptor_id: u64,
    pub block_id: ResidentBlockId,
    pub block_version: u64,
    pub referenced_bytes: usize,
    pub metadata_bytes: usize,
    pub payload_bytes_in_queue: usize,
    pub label: &'static str,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SharedQueueCompletion {
    pub descriptor_id: u64,
    pub block_id: ResidentBlockId,
    pub block_version: u64,
    pub success: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SharedQueueRejectionKind {
    QueueFull,
    QueueEmpty,
    WrongProducer,
    WrongConsumer,
    BulkPayloadInDescriptor,
    InvalidQueueBlocks,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SharedQueueRejection {
    pub kind: SharedQueueRejectionKind,
    pub descriptor_id: Option<u64>,
    pub observed_owner: ExecutionOwner,
}
