use nerva_core::types::id::DeviceOrdinal;
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;

use crate::capabilities::snapshot::CapabilityState;
use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::path::request::TransportPathRequest;
use crate::transport::path::types::{TransferMode, TransportPathClass, TransportPathKind};

#[test]
fn transport_planner_uses_verified_gpu_direct_only() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let decision = runtime
        .plan_transport_path(TransportPathRequest::new(
            MemoryTier::Vram,
            MemoryTier::Vram,
            32 * 1024,
            TransferMode::Decode,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            CapabilityState::SupportedAndVerified,
            CapabilityState::Unsupported,
            CapabilityState::SupportedUnverified,
        ))
        .unwrap();

    assert_eq!(decision.path, TransportPathKind::TrueGpuDirectRdma);
    assert_eq!(decision.class, TransportPathClass::GpuDirect);
    assert_eq!(decision.explicit_copy_bytes, 0);
    assert!(!decision.pageable_copy);
    assert!(!decision.per_token_registration);
}

#[test]
fn transport_planner_degrades_unverified_direct_path_to_pinned_host() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let decision = runtime
        .plan_transport_path(TransportPathRequest::new(
            MemoryTier::Vram,
            MemoryTier::Vram,
            32 * 1024,
            TransferMode::Decode,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            CapabilityState::SupportedUnverified,
            CapabilityState::Unsupported,
            CapabilityState::SupportedUnverified,
        ))
        .unwrap();

    assert_eq!(decision.path, TransportPathKind::OptimizedPinnedHostBounce);
    assert_eq!(decision.class, TransportPathClass::HostStaged);
    assert_eq!(decision.explicit_copy_bytes, 64 * 1024);
    assert!(!decision.pageable_copy);
    assert!(!decision.per_token_registration);
}

#[test]
fn transport_planner_can_select_mapped_pinned_for_small_decode_only() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let small_decode = runtime
        .plan_transport_path(TransportPathRequest::new(
            MemoryTier::Vram,
            MemoryTier::PinnedDram,
            32 * 1024,
            TransferMode::Decode,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            CapabilityState::Unsupported,
            CapabilityState::SupportedAndVerified,
            CapabilityState::SupportedUnverified,
        ))
        .unwrap();
    let prefill = runtime
        .plan_transport_path(TransportPathRequest::new(
            MemoryTier::Vram,
            MemoryTier::PinnedDram,
            16 * 1024 * 1024,
            TransferMode::Prefill,
            ExecutionOwner::Gpu(DeviceOrdinal(0)),
            CapabilityState::Unsupported,
            CapabilityState::SupportedAndVerified,
            CapabilityState::SupportedUnverified,
        ))
        .unwrap();

    assert_eq!(small_decode.path, TransportPathKind::MappedPinnedHostWrite);
    assert_eq!(small_decode.class, TransportPathClass::MappedPinned);
    assert_eq!(prefill.path, TransportPathKind::OptimizedPinnedHostBounce);
}
