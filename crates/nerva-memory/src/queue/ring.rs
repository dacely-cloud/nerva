use nerva_core::types::block::kind::BlockKind;
use nerva_core::types::block::residency::ResidencyState;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::ownership::coherence::CoherencePolicy;
use nerva_core::types::ownership::mutation::MutationSemantics;
use nerva_core::types::ownership::owner::ExecutionOwner;

use crate::queue::types::{
    SharedQueueCompletion, SharedQueueDescriptor, SharedQueueRejection, SharedQueueRejectionKind,
    SharedWorkQueueSpec,
};
use crate::registry::table::registry::BlockRegistry;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedWorkQueue {
    spec: SharedWorkQueueSpec,
    ring: Vec<Option<SharedQueueDescriptor>>,
    head: usize,
    tail: usize,
    len: usize,
    completions: Vec<SharedQueueCompletion>,
}

impl SharedWorkQueue {
    pub fn new(spec: SharedWorkQueueSpec, registry: &BlockRegistry) -> Result<Self> {
        if spec.capacity == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "shared work queue capacity must be non-zero".to_string(),
            });
        }
        validate_queue_block(registry, spec.descriptor_block)?;
        validate_queue_block(registry, spec.completion_block)?;
        if spec.coherence != CoherencePolicy::AtomicControlOnly {
            return Err(NervaError::InvalidArgument {
                reason: "shared work queue control blocks require AtomicControlOnly coherence"
                    .to_string(),
            });
        }
        Ok(Self {
            spec,
            ring: vec![None; spec.capacity],
            head: 0,
            tail: 0,
            len: 0,
            completions: Vec::with_capacity(spec.capacity),
        })
    }

    pub const fn capacity(&self) -> usize {
        self.spec.capacity
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn completion_count(&self) -> usize {
        self.completions.len()
    }

    pub fn post(
        &mut self,
        producer: ExecutionOwner,
        descriptor: SharedQueueDescriptor,
    ) -> core::result::Result<(), SharedQueueRejection> {
        if producer != self.spec.producer {
            return Err(rejection(
                SharedQueueRejectionKind::WrongProducer,
                Some(descriptor.descriptor_id),
                producer,
            ));
        }
        if descriptor.payload_bytes_in_queue != 0 {
            return Err(rejection(
                SharedQueueRejectionKind::BulkPayloadInDescriptor,
                Some(descriptor.descriptor_id),
                producer,
            ));
        }
        if self.len == self.spec.capacity {
            return Err(rejection(
                SharedQueueRejectionKind::QueueFull,
                Some(descriptor.descriptor_id),
                producer,
            ));
        }
        self.ring[self.tail] = Some(descriptor);
        self.tail = (self.tail + 1) % self.spec.capacity;
        self.len += 1;
        Ok(())
    }

    pub fn complete_next(
        &mut self,
        consumer: ExecutionOwner,
        success: bool,
    ) -> core::result::Result<SharedQueueCompletion, SharedQueueRejection> {
        if consumer != self.spec.consumer {
            return Err(rejection(
                SharedQueueRejectionKind::WrongConsumer,
                None,
                consumer,
            ));
        }
        if self.len == 0 {
            return Err(rejection(
                SharedQueueRejectionKind::QueueEmpty,
                None,
                consumer,
            ));
        }
        let descriptor = self.ring[self.head]
            .take()
            .expect("queue length and descriptor ring drifted");
        self.head = (self.head + 1) % self.spec.capacity;
        self.len -= 1;
        let completion = SharedQueueCompletion {
            descriptor_id: descriptor.descriptor_id,
            block_id: descriptor.block_id,
            block_version: descriptor.block_version,
            success,
        };
        self.completions.push(completion);
        Ok(completion)
    }
}

fn validate_queue_block(
    registry: &BlockRegistry,
    block_id: nerva_core::types::id::block::ResidentBlockId,
) -> Result<()> {
    let block = registry
        .block(block_id)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("shared work queue references unknown block {}", block_id.0),
        })?;
    if block.kind != BlockKind::Queue
        || block.state != ResidencyState::Ready
        || block.coherence != CoherencePolicy::AtomicControlOnly
        || block.semantics != MutationSemantics::AtomicControl
    {
        return Err(NervaError::InvalidArgument {
            reason: "shared work queue control block is not ready atomic queue metadata"
                .to_string(),
        });
    }
    Ok(())
}

fn rejection(
    kind: SharedQueueRejectionKind,
    descriptor_id: Option<u64>,
    observed_owner: ExecutionOwner,
) -> SharedQueueRejection {
    SharedQueueRejection {
        kind,
        descriptor_id,
        observed_owner,
    }
}
