use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::memory::tier::MemoryTier;
use nerva_core::types::ownership::owner::ExecutionOwner;

use crate::execution::probe::blocks::ReferenceTransactionBlocks;
use crate::execution::types::{
    ExecutionTransactionSpec, TransactionBlockUse, TransactionOperation, TransactionOperationKind,
};

pub(crate) fn reference_transaction_spec(
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
