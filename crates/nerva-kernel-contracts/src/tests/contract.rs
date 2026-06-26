use nerva_core::types::block::BlockKind;
use nerva_core::types::dtype::DType;
use nerva_core::types::memory::MemoryTier;

use crate::contract::probe::{KernelContractProbeStatus, kernel_contract_probe};
use crate::contract::types::{
    KernelBufferContract, KernelBufferRole, KernelContract, KernelContractKind, LaunchBounds,
};

#[test]
fn contract_accepts_device_resident_decode_buffer() {
    let bounds = LaunchBounds::new(64, 256).unwrap();
    let token_ring = KernelBufferContract::new(
        "device_token_ring",
        KernelBufferRole::InOut,
        BlockKind::TokenState,
        DType::U32,
        MemoryTier::Vram,
        4096,
    )
    .unwrap();
    let contract = KernelContract::new(
        "synthetic_decode",
        KernelContractKind::DecodeGraph,
        bounds,
        vec![token_ring],
    )
    .unwrap();

    assert!(contract.require_decode_ready().is_ok());
    assert_eq!(contract.buffers[0].name, "device_token_ring");
}

#[test]
fn contract_rejects_hot_path_allocation() {
    let bounds = LaunchBounds::new(1, 32).unwrap();
    let scratch = KernelBufferContract::new(
        "scratch",
        KernelBufferRole::Scratch,
        BlockKind::Workspace,
        DType::U8,
        MemoryTier::Vram,
        1024,
    )
    .unwrap();
    let contract = KernelContract::new(
        "decode_with_alloc",
        KernelContractKind::DecodeGraph,
        bounds,
        vec![scratch],
    )
    .unwrap()
    .with_hot_path_allocation_allowed(true);

    assert!(contract.require_decode_ready().is_err());
}

#[test]
fn contract_rejects_host_only_decode_buffers() {
    let bounds = LaunchBounds::new(1, 32).unwrap();
    let host_buffer = KernelBufferContract::new(
        "host_observation",
        KernelBufferRole::Output,
        BlockKind::TokenState,
        DType::U32,
        MemoryTier::Dram,
        4,
    )
    .unwrap();
    let contract = KernelContract::new(
        "host_only_decode",
        KernelContractKind::DecodeGraph,
        bounds,
        vec![host_buffer],
    )
    .unwrap();

    assert!(contract.require_decode_ready().is_err());
}

#[test]
fn launch_bounds_reject_zero_dimensions() {
    assert!(LaunchBounds::new(0, 32).is_err());
    assert!(LaunchBounds::new(1, 0).is_err());
}

#[test]
fn kernel_contract_probe_reports_decode_contract() {
    let summary = kernel_contract_probe().unwrap();

    assert_eq!(summary.status, KernelContractProbeStatus::Ok);
    assert_eq!(summary.contract_count, 1);
    assert_eq!(summary.buffer_count, 2);
    assert_eq!(summary.device_resident_buffers, 2);
    assert!(!summary.hot_path_allocation_allowed);
    assert!(summary.to_json().contains("\"status\":\"ok\""));
}
