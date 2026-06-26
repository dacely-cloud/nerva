use std::collections::VecDeque;

use nerva_core::types::error::{NervaError, Result};

use crate::transport::contract::types::{
    ReceiveDescriptor, TransferCompletion, TransferDescriptor, TransferId, TransportEndpoint,
};

#[derive(Clone, Debug)]
pub(crate) struct BoundedTransportQueues {
    receive_capacity: usize,
    completion_capacity: usize,
    posted_receives: VecDeque<PostedReceive>,
    completions: VecDeque<TransferCompletion>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct PostedReceive {
    transfer_id: TransferId,
    endpoint: TransportEndpoint,
    descriptor: ReceiveDescriptor,
}

impl PostedReceive {
    pub(crate) const fn new(
        transfer_id: TransferId,
        endpoint: TransportEndpoint,
        descriptor: ReceiveDescriptor,
    ) -> Self {
        Self {
            transfer_id,
            endpoint,
            descriptor,
        }
    }

    pub(crate) const fn descriptor(&self) -> ReceiveDescriptor {
        self.descriptor
    }
}

impl BoundedTransportQueues {
    pub(crate) fn new(receive_capacity: usize, completion_capacity: usize) -> Result<Self> {
        if receive_capacity == 0 || completion_capacity == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "transport queues must be non-zero".to_string(),
            });
        }
        Ok(Self {
            receive_capacity,
            completion_capacity,
            posted_receives: VecDeque::with_capacity(receive_capacity),
            completions: VecDeque::with_capacity(completion_capacity),
        })
    }

    pub(crate) const fn receive_capacity(&self) -> usize {
        self.receive_capacity
    }

    pub(crate) const fn completion_capacity(&self) -> usize {
        self.completion_capacity
    }

    pub(crate) fn preposted_receives(&self) -> usize {
        self.posted_receives.len()
    }

    pub(crate) fn pending_completions(&self) -> usize {
        self.completions.len()
    }

    pub(crate) fn push_receive(&mut self, receive: PostedReceive) -> Result<()> {
        if self.posted_receives.len() == self.receive_capacity {
            return Err(NervaError::AllocationFailed {
                bytes: 0,
                reason: "bounded transport receive ring is full".to_string(),
            });
        }
        self.posted_receives.push_back(receive);
        Ok(())
    }

    pub(crate) fn take_matching_receive(
        &mut self,
        dst: TransportEndpoint,
        transfer: TransferDescriptor,
    ) -> Result<PostedReceive> {
        let Some(index) = self.posted_receives.iter().position(|posted| {
            posted.endpoint == dst
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
        self.posted_receives
            .remove(index)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "matched transport receive disappeared".to_string(),
            })
    }

    pub(crate) fn ensure_completion_capacity(&self) -> Result<()> {
        if self.completions.len() == self.completion_capacity {
            return Err(NervaError::AllocationFailed {
                bytes: 0,
                reason: "bounded transport completion ring is full".to_string(),
            });
        }
        Ok(())
    }

    pub(crate) fn push_completion(&mut self, completion: TransferCompletion) -> Result<()> {
        self.ensure_completion_capacity()?;
        self.completions.push_back(completion);
        Ok(())
    }

    pub(crate) fn pop_completion(&mut self) -> Option<TransferCompletion> {
        self.completions.pop_front()
    }
}
