use nerva_core::types::block::taxonomy::BlockKind;
use nerva_core::types::id::{DeviceOrdinal, ResidentBlockId};
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::{CoherencePolicy, ExecutionOwner, MutationSemantics};

use crate::queue::probe::run::run_shared_work_queue_probe;
use crate::queue::ring::SharedWorkQueue;
use crate::queue::types::{SharedQueueDescriptor, SharedQueueRejectionKind, SharedWorkQueueSpec};
use crate::registry::request::BlockAllocationRequest;
use crate::registry::table::BlockRegistry;

#[test]
fn shared_queue_probe_reports_bounded_handle_queue() {
    let summary = run_shared_work_queue_probe().unwrap();

    assert!(summary.passed());
    assert_eq!(summary.queue_capacity, 4);
    assert_eq!(summary.queue_blocks_ready, 2);
    assert_eq!(summary.atomic_control_blocks, 2);
    assert_eq!(summary.descriptors_posted, 4);
    assert_eq!(summary.descriptors_completed, 4);
    assert_eq!(summary.completion_records, 4);
    assert_eq!(summary.queue_full_rejections, 1);
    assert_eq!(summary.wrong_producer_rejections, 1);
    assert_eq!(summary.wrong_consumer_rejections, 1);
    assert_eq!(summary.bulk_payload_rejections, 1);
    assert_eq!(summary.payload_bytes_in_queue, 0);
    assert_eq!(summary.phase_handoff_syncs, 4);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"queue_full_rejections\":1"));
    assert!(summary.to_json().contains("\"payload_bytes_in_queue\":0"));
}

#[test]
fn shared_queue_rejects_wrong_producer_and_bulk_payload() {
    let (registry, descriptor_block, completion_block, tensor_block) = queue_fixture();
    let mut queue = SharedWorkQueue::new(
        SharedWorkQueueSpec {
            capacity: 2,
            descriptor_block,
            completion_block,
            producer: ExecutionOwner::Cpu,
            consumer: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            coherence: CoherencePolicy::AtomicControlOnly,
        },
        &registry,
    )
    .unwrap();

    let valid_descriptor = descriptor(1, tensor_block, 0);
    let wrong_owner = queue
        .post(ExecutionOwner::Gpu(DeviceOrdinal(1)), valid_descriptor)
        .unwrap_err();
    assert_eq!(wrong_owner.kind, SharedQueueRejectionKind::WrongProducer);

    let bad_payload = descriptor(2, tensor_block, 64);
    let bulk_payload = queue.post(ExecutionOwner::Cpu, bad_payload).unwrap_err();
    assert_eq!(
        bulk_payload.kind,
        SharedQueueRejectionKind::BulkPayloadInDescriptor
    );
}

#[test]
fn shared_queue_rejects_overflow_and_wrong_consumer() {
    let (registry, descriptor_block, completion_block, tensor_block) = queue_fixture();
    let mut queue = SharedWorkQueue::new(
        SharedWorkQueueSpec {
            capacity: 1,
            descriptor_block,
            completion_block,
            producer: ExecutionOwner::Cpu,
            consumer: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            coherence: CoherencePolicy::AtomicControlOnly,
        },
        &registry,
    )
    .unwrap();

    queue
        .post(ExecutionOwner::Cpu, descriptor(1, tensor_block, 0))
        .unwrap();
    let full = queue
        .post(ExecutionOwner::Cpu, descriptor(2, tensor_block, 0))
        .unwrap_err();
    assert_eq!(full.kind, SharedQueueRejectionKind::QueueFull);

    let wrong_consumer = queue.complete_next(ExecutionOwner::Cpu, true).unwrap_err();
    assert_eq!(wrong_consumer.kind, SharedQueueRejectionKind::WrongConsumer);

    let completion = queue
        .complete_next(ExecutionOwner::Gpu(DeviceOrdinal(0)), true)
        .unwrap();
    assert_eq!(completion.descriptor_id, 1);
    assert_eq!(queue.completion_count(), 1);
}

#[test]
fn shared_queue_requires_ready_atomic_queue_blocks() {
    let mut registry = BlockRegistry::new([(MemoryTier::SharedHbmOrLpddr, 1024 * 1024)]);
    let descriptor_block = registry
        .allocate(BlockAllocationRequest::new(
            BlockKind::Queue,
            MemoryTier::SharedHbmOrLpddr,
            4096,
        ))
        .unwrap();
    let completion_block = ready_queue_block(&mut registry);

    let err = SharedWorkQueue::new(
        SharedWorkQueueSpec {
            capacity: 2,
            descriptor_block,
            completion_block,
            producer: ExecutionOwner::Cpu,
            consumer: ExecutionOwner::Gpu(DeviceOrdinal(0)),
            coherence: CoherencePolicy::AtomicControlOnly,
        },
        &registry,
    )
    .unwrap_err();
    assert!(format!("{err:?}").contains("queue control block"));
}

fn queue_fixture() -> (
    BlockRegistry,
    ResidentBlockId,
    ResidentBlockId,
    ResidentBlockId,
) {
    let mut registry = BlockRegistry::new([
        (MemoryTier::SharedHbmOrLpddr, 1024 * 1024),
        (MemoryTier::Vram, 1024 * 1024),
    ]);
    let descriptor_block = ready_queue_block(&mut registry);
    let completion_block = ready_queue_block(&mut registry);
    let tensor_block = registry
        .allocate(BlockAllocationRequest::new(
            BlockKind::Activation,
            MemoryTier::Vram,
            4096,
        ))
        .unwrap();
    {
        let block = registry.block_mut(tensor_block).unwrap();
        block.owner = ExecutionOwner::Gpu(DeviceOrdinal(0));
        block.version = 1;
    }
    registry.mark_ready(tensor_block).unwrap();
    (registry, descriptor_block, completion_block, tensor_block)
}

fn ready_queue_block(registry: &mut BlockRegistry) -> ResidentBlockId {
    let block_id = registry
        .allocate(BlockAllocationRequest::new(
            BlockKind::Queue,
            MemoryTier::SharedHbmOrLpddr,
            4096,
        ))
        .unwrap();
    {
        let block = registry.block_mut(block_id).unwrap();
        block.coherence = CoherencePolicy::AtomicControlOnly;
        block.semantics = MutationSemantics::AtomicControl;
    }
    registry.mark_ready(block_id).unwrap();
    block_id
}

fn descriptor(
    descriptor_id: u64,
    block_id: ResidentBlockId,
    payload_bytes_in_queue: usize,
) -> SharedQueueDescriptor {
    SharedQueueDescriptor {
        descriptor_id,
        block_id,
        block_version: 1,
        referenced_bytes: 4096,
        metadata_bytes: core::mem::size_of::<SharedQueueDescriptor>(),
        payload_bytes_in_queue,
        label: "test_descriptor",
    }
}
