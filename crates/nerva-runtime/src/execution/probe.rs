use nerva_core::types::block::BlockKind;
use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::{DeviceOrdinal, LayoutId};
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_memory::registry::{BlockAllocationRequest, BlockRegistry};

use crate::execution::plan::plan_execution_transaction;
use crate::execution::summary::ExecutionTransactionSummary;
use crate::execution::types::{
    ExecutionTransactionSpec, TransactionBlockUse, TransactionOperation, TransactionOperationKind,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ReferenceTransactionBlocks {
    pub device_token: nerva_core::types::id::ResidentBlockId,
    pub weight_tile: nerva_core::types::id::ResidentBlockId,
    pub kv_page: nerva_core::types::id::ResidentBlockId,
    pub logits: nerva_core::types::id::ResidentBlockId,
    pub host_token: nerva_core::types::id::ResidentBlockId,
}

pub fn run_execution_transaction_probe(
    device: DeviceOrdinal,
) -> Result<ExecutionTransactionSummary> {
    let (registry, spec, _) = reference_transaction_fixture(device)?;
    let plan = plan_execution_transaction(spec, &registry)?;
    Ok(plan.summary)
}

pub(crate) fn reference_transaction_fixture(
    device: DeviceOrdinal,
) -> Result<(
    BlockRegistry,
    ExecutionTransactionSpec,
    ReferenceTransactionBlocks,
)> {
    let gpu = ExecutionOwner::Gpu(device);
    let mut registry = BlockRegistry::new([
        (MemoryTier::Vram, 16 * 1024 * 1024),
        (MemoryTier::PinnedDram, 1024 * 1024),
        (MemoryTier::Dram, 16 * 1024 * 1024),
    ]);

    let device_token = allocate_ready_block(
        &mut registry,
        BlockKind::TokenState,
        MemoryTier::Vram,
        DType::U32,
        4096,
        gpu,
    )?;
    let weight_tile = allocate_ready_block(
        &mut registry,
        BlockKind::Weight,
        MemoryTier::Vram,
        DType::F16,
        128 * 1024,
        ExecutionOwner::SharedReadOnly,
    )?;
    let kv_page = allocate_ready_block(
        &mut registry,
        BlockKind::KvPage,
        MemoryTier::Vram,
        DType::F16,
        64 * 1024,
        gpu,
    )?;
    let logits = allocate_ready_block(
        &mut registry,
        BlockKind::Logits,
        MemoryTier::Vram,
        DType::F32,
        32 * 1024,
        ExecutionOwner::Cpu,
    )?;
    let host_token = registry.allocate(
        BlockAllocationRequest::new(BlockKind::TokenState, MemoryTier::PinnedDram, 4096)
            .with_dtype(DType::U32)
            .with_layout(LayoutId(1)),
    )?;
    registry.mark_ready(host_token)?;

    let blocks = ReferenceTransactionBlocks {
        device_token,
        weight_tile,
        kv_page,
        logits,
        host_token,
    };
    let spec = reference_transaction_spec(device, blocks);
    Ok((registry, spec, blocks))
}

fn reference_transaction_spec(
    device: DeviceOrdinal,
    blocks: ReferenceTransactionBlocks,
) -> ExecutionTransactionSpec {
    let gpu = ExecutionOwner::Gpu(device);
    ExecutionTransactionSpec::new("reference_decode_transaction", 0)
        .with_operation(
            TransactionOperation::new(
                "decode_tensor_step",
                TransactionOperationKind::TensorCompute,
                gpu,
                1_200,
            )
            .graph_capturable(true)
            .with_block_use(TransactionBlockUse::read(
                blocks.device_token,
                gpu,
                MemoryTier::Vram,
                1,
                "decode_reads_device_token",
            ))
            .with_block_use(TransactionBlockUse::read(
                blocks.weight_tile,
                ExecutionOwner::SharedReadOnly,
                MemoryTier::Vram,
                1,
                "decode_reads_weight_tile",
            ))
            .with_block_use(TransactionBlockUse::read(
                blocks.kv_page,
                gpu,
                MemoryTier::Vram,
                1,
                "decode_reads_kv_page",
            ))
            .with_block_use(TransactionBlockUse::write(
                blocks.logits,
                gpu,
                MemoryTier::Vram,
                1,
                "decode_writes_logits",
            )),
        )
        .with_operation(
            TransactionOperation::new("kv_append", TransactionOperationKind::KvAppend, gpu, 400)
                .graph_capturable(true)
                .with_block_use(TransactionBlockUse::read(
                    blocks.device_token,
                    gpu,
                    MemoryTier::Vram,
                    1,
                    "kv_append_reads_device_token",
                ))
                .with_block_use(TransactionBlockUse::read_write(
                    blocks.kv_page,
                    gpu,
                    MemoryTier::Vram,
                    1,
                    "kv_append_updates_kv_page",
                )),
        )
        .with_operation(
            TransactionOperation::new(
                "device_greedy_sample",
                TransactionOperationKind::DeviceSampling,
                gpu,
                250,
            )
            .graph_capturable(true)
            .with_block_use(TransactionBlockUse::read(
                blocks.logits,
                gpu,
                MemoryTier::Vram,
                1,
                "sampler_reads_logits",
            ))
            .with_block_use(TransactionBlockUse::write(
                blocks.device_token,
                gpu,
                MemoryTier::Vram,
                1,
                "sampler_writes_device_token",
            )),
        )
        .with_operation(
            TransactionOperation::new(
                "host_token_observe",
                TransactionOperationKind::HostObservation,
                ExecutionOwner::Cpu,
                125,
            )
            .with_block_use(TransactionBlockUse::read(
                blocks.device_token,
                gpu,
                MemoryTier::Vram,
                1,
                "host_observer_reads_device_token_replica",
            ))
            .with_block_use(TransactionBlockUse::write(
                blocks.host_token,
                ExecutionOwner::Cpu,
                MemoryTier::PinnedDram,
                0,
                "host_observer_writes_pinned_token",
            )),
        )
}

fn allocate_ready_block(
    registry: &mut BlockRegistry,
    kind: BlockKind,
    tier: MemoryTier,
    dtype: DType,
    bytes: usize,
    owner: ExecutionOwner,
) -> Result<nerva_core::types::id::ResidentBlockId> {
    let id = registry.allocate(
        BlockAllocationRequest::new(kind, tier, bytes)
            .with_dtype(dtype)
            .with_layout(LayoutId(1)),
    )?;
    let block = registry
        .block_mut(id)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("allocated block {} disappeared", id.0),
        })?;
    block.publish(owner);
    Ok(id)
}
