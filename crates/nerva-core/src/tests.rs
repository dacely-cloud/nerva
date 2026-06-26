use crate::types::{
    BlockKind, BlockShape, DeviceOrdinal, ExecutionOwner, HostArch, MemoryDomainId, MemoryTier,
    MutationSemantics, NervaError, ReplicaId, ResidentBlock, ResidentBlockId,
    ensure_supported_linux_host, host_arch,
};

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
