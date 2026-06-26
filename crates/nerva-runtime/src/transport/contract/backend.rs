use nerva_core::types::block::resident::ResidentBlock;
use nerva_core::types::error::Result;
use nerva_core::types::id::replica::ReplicaId;

use crate::transport::contract::backend::queues::{BoundedTransportQueues, PostedReceive};
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

mod queues;
mod validate;

#[derive(Clone, Debug)]
pub struct PinnedHostLoopbackTransport {
    backend: TransportRegistrationBackend,
    cache: TransportRegistrationCache,
    next_transfer_id: u64,
    queues: BoundedTransportQueues,
}

impl PinnedHostLoopbackTransport {
    pub fn new(
        registration_capacity: usize,
        receive_capacity: usize,
        completion_capacity: usize,
    ) -> Result<Self> {
        Ok(Self {
            backend: TransportRegistrationBackend::RdmaPinnedHost,
            cache: TransportRegistrationCache::new(registration_capacity)?,
            next_transfer_id: 1,
            queues: BoundedTransportQueues::new(receive_capacity, completion_capacity)?,
        })
    }

    pub const fn registration_backend(&self) -> TransportRegistrationBackend {
        self.backend
    }

    pub fn registered_entries(&self) -> usize {
        self.cache.len()
    }

    pub fn preposted_receives(&self) -> usize {
        self.queues.preposted_receives()
    }

    pub const fn receive_queue_capacity(&self) -> usize {
        self.queues.receive_capacity()
    }

    pub const fn completion_queue_capacity(&self) -> usize {
        self.queues.completion_capacity()
    }

    pub fn pending_completions(&self) -> usize {
        self.queues.pending_completions()
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
        let transfer_id = self.next_id();
        self.queues
            .push_receive(PostedReceive::new(transfer_id, *src, receive))?;
        Ok(transfer_id)
    }

    fn send(&mut self, dst: &Self::Endpoint, transfer: TransferDescriptor) -> Result<TransferId> {
        validate_transfer_descriptor(self.backend, transfer)?;
        self.queues.ensure_completion_capacity()?;
        let receive = self.queues.take_matching_receive(*dst, transfer)?;
        let transfer_id = self.next_id();
        self.queues.push_completion(TransferCompletion {
            transfer_id,
            source_block: transfer.source.key.block_id,
            destination_block: receive.descriptor().destination.key.block_id,
            block_version: transfer.block_version,
            bytes: transfer.bytes,
            mode: transfer.mode,
            status: TransferCompletionStatus::Complete,
        })?;
        Ok(transfer_id)
    }

    fn poll(&mut self, completions: &mut [TransferCompletion]) -> Result<usize> {
        let mut copied = 0;
        for slot in completions.iter_mut() {
            let Some(completion) = self.queues.pop_completion() else {
                break;
            };
            *slot = completion;
            copied += 1;
        }
        Ok(copied)
    }
}
