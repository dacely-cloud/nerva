use nerva_core::types::block::taxonomy::BlockKind;
use nerva_core::types::error::Result;
use nerva_core::types::id::{DeviceOrdinal, ResidentBlockId};
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::{CoherencePolicy, ExecutionOwner, MutationSemantics};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::TokenLedger;

use crate::queue::ring::SharedWorkQueue;
use crate::queue::summary::{SharedQueueProbeStatus, SharedQueueProbeSummary};
use crate::queue::types::{SharedQueueDescriptor, SharedQueueRejectionKind, SharedWorkQueueSpec};
use crate::registry::{BlockAllocationRequest, BlockRegistry};

pub fn run_shared_work_queue_probe() -> Result<SharedQueueProbeSummary> {
    let mut registry = BlockRegistry::new([
        (MemoryTier::SharedHbmOrLpddr, 1024 * 1024),
        (MemoryTier::Vram, 1024 * 1024),
    ]);
    let descriptor_block = allocate_queue_block(&mut registry)?;
    let completion_block = allocate_queue_block(&mut registry)?;
    let tensor_block = allocate_tensor_block(&mut registry, 4096)?;

    let producer = ExecutionOwner::Cpu;
    let consumer = ExecutionOwner::Gpu(DeviceOrdinal(0));
    let mut queue = SharedWorkQueue::new(
        SharedWorkQueueSpec {
            capacity: 4,
            descriptor_block,
            completion_block,
            producer,
            consumer,
            coherence: CoherencePolicy::AtomicControlOnly,
        },
        &registry,
    )?;

    let mut counters = SharedQueueCounters::new();
    let bad_bulk = descriptor(99, tensor_block, 1, 4096, 128);
    counters.record_rejection(queue.post(producer, bad_bulk).unwrap_err().kind);
    let wrong_producer = descriptor(98, tensor_block, 1, 4096, 0);
    counters.record_rejection(
        queue
            .post(ExecutionOwner::Gpu(DeviceOrdinal(1)), wrong_producer)
            .unwrap_err()
            .kind,
    );

    for descriptor_id in 0..queue.capacity() as u64 {
        let descriptor = descriptor(descriptor_id, tensor_block, 1, 4096, 0);
        queue
            .post(producer, descriptor)
            .expect("queue has capacity");
        counters.descriptors_posted += 1;
        counters.referenced_block_bytes = counters
            .referenced_block_bytes
            .saturating_add(descriptor.referenced_bytes as u64);
        counters.payload_bytes_in_queue = counters
            .payload_bytes_in_queue
            .saturating_add(descriptor.payload_bytes_in_queue as u64);
    }

    let overflow = descriptor(100, tensor_block, 1, 4096, 0);
    counters.record_rejection(queue.post(producer, overflow).unwrap_err().kind);
    counters.record_rejection(
        queue
            .complete_next(ExecutionOwner::Cpu, true)
            .unwrap_err()
            .kind,
    );

    let mut ledger = TokenLedger::new(0);
    while queue.len() > 0 {
        let completion = queue
            .complete_next(consumer, true)
            .expect("consumer owns queue");
        counters.descriptors_completed += 1;
        ledger.record_sync(
            SyncClass::PhaseHandoff,
            Some(completion.block_id),
            Some(MemoryTier::SharedHbmOrLpddr),
            Some(MemoryTier::Vram),
            0,
            1,
            MetricSource::EstimatedModel,
            "shared_queue_descriptor_completion",
        );
    }
    ledger.require_classified_syncs()?;
    ledger.require_zero_hot_path_allocations()?;

    Ok(SharedQueueProbeSummary {
        status: SharedQueueProbeStatus::Ok,
        queue_capacity: queue.capacity() as u64,
        queue_blocks_ready: count_queue_blocks(&registry, [descriptor_block, completion_block]),
        atomic_control_blocks: count_atomic_control_blocks(
            &registry,
            [descriptor_block, completion_block],
        ),
        descriptors_posted: counters.descriptors_posted,
        descriptors_completed: counters.descriptors_completed,
        completion_records: queue.completion_count() as u64,
        queue_full_rejections: counters.queue_full_rejections,
        wrong_producer_rejections: counters.wrong_producer_rejections,
        wrong_consumer_rejections: counters.wrong_consumer_rejections,
        bulk_payload_rejections: counters.bulk_payload_rejections,
        payload_bytes_in_queue: counters.payload_bytes_in_queue,
        referenced_block_bytes: counters.referenced_block_bytes,
        phase_handoff_syncs: ledger.sync_count_for(SyncClass::PhaseHandoff),
        hot_path_allocations: ledger.hot_path_allocations,
        error: None,
    })
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct SharedQueueCounters {
    descriptors_posted: u64,
    descriptors_completed: u64,
    queue_full_rejections: u64,
    wrong_producer_rejections: u64,
    wrong_consumer_rejections: u64,
    bulk_payload_rejections: u64,
    payload_bytes_in_queue: u64,
    referenced_block_bytes: u64,
}

impl SharedQueueCounters {
    const fn new() -> Self {
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

    fn record_rejection(&mut self, kind: SharedQueueRejectionKind) {
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

fn allocate_queue_block(registry: &mut BlockRegistry) -> Result<ResidentBlockId> {
    let block_id = registry.allocate(BlockAllocationRequest::new(
        BlockKind::Queue,
        MemoryTier::SharedHbmOrLpddr,
        4096,
    ))?;
    {
        let block = registry
            .block_mut(block_id)
            .expect("allocated block exists");
        block.coherence = CoherencePolicy::AtomicControlOnly;
        block.semantics = MutationSemantics::AtomicControl;
        block.owner = ExecutionOwner::PhaseTransition;
    }
    registry.mark_ready(block_id)?;
    Ok(block_id)
}

fn allocate_tensor_block(registry: &mut BlockRegistry, bytes: usize) -> Result<ResidentBlockId> {
    let block_id = registry.allocate(BlockAllocationRequest::new(
        BlockKind::Activation,
        MemoryTier::Vram,
        bytes,
    ))?;
    {
        let block = registry
            .block_mut(block_id)
            .expect("allocated block exists");
        block.owner = ExecutionOwner::Gpu(DeviceOrdinal(0));
        block.version = 1;
    }
    registry.mark_ready(block_id)?;
    Ok(block_id)
}

fn descriptor(
    descriptor_id: u64,
    block_id: ResidentBlockId,
    block_version: u64,
    referenced_bytes: usize,
    payload_bytes_in_queue: usize,
) -> SharedQueueDescriptor {
    SharedQueueDescriptor {
        descriptor_id,
        block_id,
        block_version,
        referenced_bytes,
        metadata_bytes: core::mem::size_of::<SharedQueueDescriptor>(),
        payload_bytes_in_queue,
        label: "shared_queue_block_handle",
    }
}

fn count_queue_blocks(registry: &BlockRegistry, block_ids: [ResidentBlockId; 2]) -> u64 {
    block_ids
        .iter()
        .filter(|block_id| {
            registry
                .block(**block_id)
                .is_some_and(|block| block.kind == BlockKind::Queue)
        })
        .count() as u64
}

fn count_atomic_control_blocks(registry: &BlockRegistry, block_ids: [ResidentBlockId; 2]) -> u64 {
    block_ids
        .iter()
        .filter(|block_id| {
            registry.block(**block_id).is_some_and(|block| {
                block.coherence == CoherencePolicy::AtomicControlOnly
                    && block.semantics == MutationSemantics::AtomicControl
            })
        })
        .count() as u64
}
