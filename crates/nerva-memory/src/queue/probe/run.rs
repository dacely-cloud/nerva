use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::coherence::CoherencePolicy;
use nerva_core::types::ownership::owner::ExecutionOwner;

use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::queue::probe::counters::SharedQueueCounters;
use crate::queue::probe::fixtures::{
    allocate_queue_block, allocate_tensor_block, count_atomic_control_blocks, count_queue_blocks,
    descriptor,
};
use crate::queue::ring::SharedWorkQueue;
use crate::queue::summary::{SharedQueueProbeStatus, SharedQueueProbeSummary};
use crate::queue::types::SharedWorkQueueSpec;
use crate::registry::table::registry::BlockRegistry;

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
