use crate::types::arch::{HostArch, ensure_supported_linux_host, host_arch};
use crate::types::backend::capabilities::{
    BackendArchitecture, DeviceBackendCapabilities, DeviceBackendKind,
};
use crate::types::backend::operation::{
    BackendAllocationContract, BackendGraphExecContract, BackendQueueContract,
    BackendTransactionDescriptor,
};
use crate::types::backend::validation::validate_backend_contract;
use crate::types::block::resident::ResidentBlock;
use crate::types::block::taxonomy::BlockKind;
use crate::types::dtype::DType;
use crate::types::error::NervaError;
use crate::types::id::{DeviceOrdinal, MemoryDomainId, ReplicaId, ResidentBlockId, TransactionId};
use crate::types::memory::{MemoryFabricKind, MemoryTier};
use crate::types::ownership::{ExecutionOwner, MutationSemantics};
use crate::types::shape::BlockShape;

#[test]
fn linux_host_gate_accepts_build_host_or_reports_other() {
    let result = ensure_supported_linux_host();
    if matches!(host_arch(), HostArch::Other) {
        assert!(matches!(result, Err(NervaError::UnsupportedHost { .. })));
    } else {
        assert!(result.is_ok());
    }
}

#[test]
fn resident_block_carries_identity_and_tier() {
    let block = ResidentBlock::new(
        ResidentBlockId(7),
        BlockKind::KvPage,
        MemoryTier::Vram,
        4096,
    );
    assert_eq!(block.id, ResidentBlockId(7));
    assert_eq!(block.tier, MemoryTier::Vram);
    assert_eq!(block.memory_domain, MemoryDomainId::GPU_VRAM);
    assert_eq!(block.semantics, MutationSemantics::AppendOnly);
}

#[test]
fn ready_check_rejects_stale_or_absent_replicas() {
    let mut block = ResidentBlock::new(
        ResidentBlockId(11),
        BlockKind::TokenState,
        MemoryTier::Vram,
        64,
    );
    let replica = block.authoritative_copy;

    assert!(block.require_ready(replica, 0).is_err());
    block.mark_ready();
    assert!(block.require_ready(ReplicaId(999), 0).is_err());
    assert!(block.require_ready(replica, 1).is_err());

    let version = block.publish(ExecutionOwner::Gpu(DeviceOrdinal(0)));
    assert_eq!(version, 1);
    assert!(block.require_ready(replica, 1).is_ok());
}

#[test]
fn block_shape_rejects_zero_dimensions() {
    assert_eq!(BlockShape::from_dims([2, 4]).unwrap().dims(), &[2, 4]);
    assert!(BlockShape::from_dims([2, 0]).is_err());
}

#[test]
fn backend_contract_validation_requires_real_bootstrap_surfaces() {
    let capabilities = DeviceBackendCapabilities {
        kind: DeviceBackendKind::Cuda,
        device: DeviceOrdinal(0),
        name: Some("device".to_string()),
        architecture: Some(BackendArchitecture {
            major: 12,
            minor: 0,
        }),
        fabric: MemoryFabricKind::DiscreteExplicit,
        total_device_memory_bytes: Some(32 * 1024 * 1024 * 1024),
        supports_device_allocations: true,
        supports_pinned_host_allocations: true,
        supports_streams: true,
        supports_events: true,
        supports_graph_capture: true,
        supports_async_copies: true,
        supports_device_sampling: true,
        exact_dtypes: vec![DType::F16],
    };
    let device_allocation = BackendAllocationContract {
        tier: MemoryTier::Vram,
        bytes: 4096,
        alignment: 256,
        preallocated: true,
    };
    let pinned_allocation = BackendAllocationContract {
        tier: MemoryTier::PinnedDram,
        bytes: 4096,
        alignment: 64,
        preallocated: true,
    };
    let queue = BackendQueueContract {
        device: DeviceOrdinal(0),
        bounded: true,
        stream_ordered: true,
        preallocated: true,
    };
    let graph = BackendGraphExecContract {
        transaction: BackendTransactionDescriptor {
            id: TransactionId(1),
            operation_count: 2,
            block_use_count: 4,
            graph_capturable: true,
        },
        replayable: true,
    };

    let validation = validate_backend_contract(
        &capabilities,
        device_allocation,
        pinned_allocation,
        queue,
        graph,
    );

    assert!(capabilities.supports_exact_dtype(DType::F16));
    assert!(!capabilities.supports_exact_dtype(DType::BF16));
    assert!(validation.passed());
}

#[test]
fn backend_contract_validation_rejects_unbounded_or_host_only_surfaces() {
    let capabilities = DeviceBackendCapabilities {
        kind: DeviceBackendKind::Cuda,
        device: DeviceOrdinal(0),
        name: None,
        architecture: None,
        fabric: MemoryFabricKind::DiscreteExplicit,
        total_device_memory_bytes: None,
        supports_device_allocations: true,
        supports_pinned_host_allocations: false,
        supports_streams: true,
        supports_events: true,
        supports_graph_capture: true,
        supports_async_copies: true,
        supports_device_sampling: true,
        exact_dtypes: vec![DType::F16],
    };
    let validation = validate_backend_contract(
        &capabilities,
        BackendAllocationContract {
            tier: MemoryTier::Dram,
            bytes: 4096,
            alignment: 256,
            preallocated: true,
        },
        BackendAllocationContract {
            tier: MemoryTier::PinnedDram,
            bytes: 4096,
            alignment: 64,
            preallocated: true,
        },
        BackendQueueContract {
            device: DeviceOrdinal(0),
            bounded: false,
            stream_ordered: true,
            preallocated: true,
        },
        BackendGraphExecContract {
            transaction: BackendTransactionDescriptor {
                id: TransactionId(1),
                operation_count: 2,
                block_use_count: 4,
                graph_capturable: true,
            },
            replayable: true,
        },
    );

    assert!(!validation.bootstrap_decode_ready);
    assert!(!validation.device_allocation_ready);
    assert!(!validation.queue_ready);
    assert!(!validation.passed());
}
