use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::replica::ReplicaId;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::transport::TransportDeviceId;

use crate::transport::path::types::TransferMode;
use crate::transport::registration::types::TransportRegistration;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransportEndpoint {
    pub device: TransportDeviceId,
    pub stage_id: u32,
    pub lane_id: u32,
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct TransferId(pub u64);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransferDescriptor {
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub source: TransportRegistration,
    pub source_replica: ReplicaId,
    pub source_offset: usize,
    pub block_version: u64,
    pub bytes: usize,
    pub mode: TransferMode,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ReceiveDescriptor {
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub destination: TransportRegistration,
    pub destination_replica: ReplicaId,
    pub destination_offset: usize,
    pub expected_source_block: ResidentBlockId,
    pub expected_version: u64,
    pub bytes: usize,
    pub mode: TransferMode,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TransferCompletionStatus {
    Complete,
}

impl TransferCompletionStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TransferCompletion {
    pub transfer_id: TransferId,
    pub source_block: ResidentBlockId,
    pub destination_block: ResidentBlockId,
    pub block_version: u64,
    pub bytes: usize,
    pub mode: TransferMode,
    pub status: TransferCompletionStatus,
}
