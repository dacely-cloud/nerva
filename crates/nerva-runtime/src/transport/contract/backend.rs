use std::collections::VecDeque;

use nerva_core::types::block::resident::ResidentBlock;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::replica::ReplicaId;

use crate::transport::contract::backend::validate::{
    validate_receive_descriptor, validate_transfer_descriptor,
};
use crate::transport::contract::traits::TensorTransportContract;
use crate::transport::contract::types::{
    ReceiveDescriptor, TransferCompletion, TransferCompletionStatus, TransferDescriptor,
    TransferId, TransportEndpoint,
};
use crate::transport::registration::cache::TransportRegistrationCache;
use crate::transport::registration::types::{TransportRegistration, TransportRegistrationBackend};

mod validate;

#[derive(Clone, Debug)]
pub struct PinnedHostLoopbackTransport {
    backend: TransportRegistrationBackend,
    cache: TransportRegistrationCache,
    next_transfer_id: u64,
    receive_capacity: usize,
    completion_capacity: usize,
    posted_receives: VecDeque<PostedReceive>,
    completions: VecDeque<TransferCompletion>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct PostedReceive {
    transfer_id: TransferId,
    endpoint: TransportEndpoint,
    descriptor: ReceiveDescriptor,
}

impl PinnedHostLoopbackTransport {
    pub fn new(
        registration_capacity: usize,
        receive_capacity: usize,
        completion_capacity: usize,
    ) -> Result<Self> {
        if receive_capacity == 0 || completion_capacity == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "transport queues must be non-zero".to_string(),
            });
        }
        Ok(Self {
            backend: TransportRegistrationBackend::RdmaPinnedHost,
            cache: TransportRegistrationCache::new(registration_capacity)?,
            next_transfer_id: 1,
            receive_capacity,
            completion_capacity,
            posted_receives: VecDeque::with_capacity(receive_capacity),
            completions: VecDeque::with_capacity(completion_capacity),
        })
    }

    pub const fn registration_backend(&self) -> TransportRegistrationBackend {
        self.backend
    }

    pub fn registered_entries(&self) -> usize {
        self.cache.len()
    }

    pub fn preposted_receives(&self) -> usize {
        self.posted_receives.len()
    }

    pub const fn receive_queue_capacity(&self) -> usize {
        self.receive_capacity
    }

    pub const fn completion_queue_capacity(&self) -> usize {
        self.completion_capacity
    }

    pub fn pending_completions(&self) -> usize {
        self.completions.len()
    }

    fn next_id(&mut self) -> TransferId {
        let id = TransferId(self.next_transfer_id);
        self.next_transfer_id = self.next_transfer_id.saturating_add(1);
        id
    }
}

impl TensorTransportContract for PinnedHostLoopbackTransport {
    type Endpoint = TransportEndpoint;
    type Registration = TransportRegistration;

    fn register(
        &mut self,
        block: &ResidentBlock,
        replica: ReplicaId,
    ) -> Result<Self::Registration> {
        self.cache.register(block, replica, self.backend)
    }

    fn post_receive(
        &mut self,
        src: &Self::Endpoint,
        receive: ReceiveDescriptor,
    ) -> Result<TransferId> {
        validate_receive_descriptor(self.backend, receive)?;
        if self.posted_receives.len() == self.receive_capacity {
            return Err(NervaError::AllocationFailed {
                bytes: 0,
                reason: "bounded transport receive ring is full".to_string(),
            });
        }
        let transfer_id = self.next_id();
        self.posted_receives.push_back(PostedReceive {
            transfer_id,
            endpoint: *src,
            descriptor: receive,
        });
        Ok(transfer_id)
    }

    fn send(&mut self, dst: &Self::Endpoint, transfer: TransferDescriptor) -> Result<TransferId> {
        validate_transfer_descriptor(self.backend, transfer)?;
        if self.completions.len() == self.completion_capacity {
            return Err(NervaError::AllocationFailed {
                bytes: 0,
                reason: "bounded transport completion ring is full".to_string(),
            });
        }
        let Some(index) = self.posted_receives.iter().position(|posted| {
            posted.endpoint == *dst
                && posted.descriptor.request_id == transfer.request_id
                && posted.descriptor.sequence_id == transfer.sequence_id
                && posted.descriptor.expected_source_block == transfer.source.key.block_id
                && posted.descriptor.expected_version == transfer.block_version
                && posted.descriptor.bytes == transfer.bytes
                && posted.descriptor.mode == transfer.mode
        }) else {
            return Err(NervaError::InvalidArgument {
                reason: "transport send requires a matching preposted receive".to_string(),
            });
        };
        let receive = self
            .posted_receives
            .remove(index)
            .expect("matched receive exists");
        let transfer_id = self.next_id();
        self.completions.push_back(TransferCompletion {
            transfer_id,
            source_block: transfer.source.key.block_id,
            destination_block: receive.descriptor.destination.key.block_id,
            block_version: transfer.block_version,
            bytes: transfer.bytes,
            mode: transfer.mode,
            status: TransferCompletionStatus::Complete,
        });
        Ok(transfer_id)
    }

    fn poll(&mut self, completions: &mut [TransferCompletion]) -> Result<usize> {
        let mut copied = 0;
        for slot in completions.iter_mut() {
            let Some(completion) = self.completions.pop_front() else {
                break;
            };
            *slot = completion;
            copied += 1;
        }
        Ok(copied)
    }
}
