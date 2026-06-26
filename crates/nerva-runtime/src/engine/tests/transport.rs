use nerva_core::types::id::DeviceOrdinal;
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;

use crate::capabilities::snapshot::CapabilityState;
use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::transport::matrix::types::TransportCapabilityMatrixStatus;
use crate::transport::path::{
    TransferMode, TransportPathClass, TransportPathKind, TransportPathRequest,
};
use crate::transport::probe::TransportPathProbeStatus;
use crate::transport::stage::config::StagePipelineConfig;
use crate::transport::stage::summary::StagePipelineStatus;

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

#[test]
fn transport_path_probe_reports_explicit_fallback_without_hot_allocations() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_transport_path_probe().unwrap();

    assert_eq!(summary.status, TransportPathProbeStatus::Ok);
    assert_eq!(summary.requests, 7);
    assert_eq!(summary.decode_requests, 4);
    assert_eq!(summary.prefill_requests, 3);
    assert_eq!(summary.pinned_host_paths, 6);
    assert_eq!(summary.cpu_produced_paths, 1);
    assert_eq!(summary.transport_events, 7);
    assert_eq!(summary.copy_events, 6);
    assert_eq!(summary.sync_events, 7);
    assert_eq!(summary.phase_handoff_syncs, 7);
    assert_eq!(summary.fallback_decisions, 6);
    assert_eq!(summary.estimated_events, 20);
    assert_eq!(summary.estimated_latency_ns, summary.total_latency_ns);
    assert_eq!(summary.pageable_copies, 0);
    assert_eq!(summary.per_token_registrations, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.explicit_copy_bytes > 0);
    assert!(summary.to_json().contains("\"pinned_host_paths\":6"));
}

#[test]
fn transport_capability_matrix_reports_required_sizes_and_degradation() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_transport_capability_matrix_probe().unwrap();

    assert_eq!(summary.status, TransportCapabilityMatrixStatus::Ok);
    assert_eq!(summary.sizes, 6);
    assert_eq!(summary.entries.len(), 24);
    assert_eq!(summary.decode_entries, 12);
    assert_eq!(summary.prefill_entries, 12);
    assert_eq!(summary.host_staged_entries, 18);
    assert_eq!(summary.cpu_produced_entries, 6);
    assert_eq!(summary.gpu_direct_entries, 0);
    assert_eq!(summary.mapped_pinned_entries, 0);
    assert_eq!(summary.degraded_to_pinned_host_entries, 12);
    assert_eq!(summary.supported_unverified_entries, 6);
    assert_eq!(summary.supported_verified_entries, 6);
    assert_eq!(summary.unsupported_entries, 0);
    assert_eq!(summary.pageable_copies, 0);
    assert_eq!(summary.per_token_registrations, 0);
    assert_eq!(
        summary.registration_cache_hits,
        summary.entries.len() as u64
    );
    assert_eq!(summary.credit_stall_ns, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.explicit_copy_bytes > 0);
    assert!(summary.estimated_cpu_core_ns > 0);
    assert!(summary.dram_read_bytes > 0);
    assert!(summary.dram_write_bytes > 0);
    assert!(summary.pcie_tx_bytes > 0);
    assert!(summary.pcie_rx_bytes > 0);
    assert!(summary.total_estimated_visible_ns > 0);
    assert!(summary.p50_estimated_visible_ns > 0);
    assert!(summary.p95_estimated_visible_ns >= summary.p50_estimated_visible_ns);
    assert!(summary.p99_estimated_visible_ns >= summary.p95_estimated_visible_ns);
    assert!(
        summary
            .entries
            .iter()
            .all(|entry| entry.effective_payload_bandwidth_bps > 0)
    );
    assert!(summary.entries.iter().all(|entry| entry.queue_depth > 0));
    assert!(
        summary
            .entries
            .iter()
            .all(|entry| entry.registration_cache_hit)
    );
    let json = summary.to_json();
    assert!(json.contains("\"requested_path\":\"A_GPU_DIRECT_RDMA\""));
    assert!(json.contains("\"size_bytes\":32768"));
    assert!(json.contains("\"capability_result\":\"DEGRADED_TO_PINNED_HOST\""));
    assert!(json.contains("\"metric_source\":\"estimated_model\""));
    assert!(json.contains("\"p95_estimated_visible_ns\""));
    assert!(json.contains("\"effective_payload_bandwidth_bps\""));
    assert!(json.contains("\"estimated_cpu_core_ns\""));
    assert!(json.contains("\"dram_read_bytes\""));
    assert!(json.contains("\"dram_write_bytes\""));
    assert!(json.contains("\"pcie_tx_bytes\""));
    assert!(json.contains("\"pcie_rx_bytes\""));
    assert!(json.contains("\"registration_cache_hits\""));
    assert!(json.contains("\"credit_stall_ns\""));
}

#[test]
fn stage_pipeline_probe_moves_activations_without_moving_weights() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime
        .run_stage_pipeline_probe(StagePipelineConfig::reference_decode())
        .unwrap();

    assert_eq!(summary.status, StagePipelineStatus::Ok);
    assert_eq!(summary.stages, 4);
    assert_eq!(summary.boundaries, 3);
    assert_eq!(summary.activation_bytes_per_boundary, 32 * 1024);
    assert_eq!(summary.total_activation_tx_bytes, 96 * 1024);
    assert_eq!(summary.activation_only_boundaries, 3);
    assert_eq!(summary.inter_stage_weight_bytes, 0);
    assert_eq!(summary.all_reduce_bytes, 0);
    assert_eq!(summary.transport_events, 3);
    assert_eq!(summary.phase_handoff_syncs, 3);
    assert_eq!(summary.pageable_copies, 0);
    assert_eq!(summary.per_token_registrations, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.stage_local_weight_bytes > 0);
    assert!(summary.stage_local_kv_bytes > 0);
    assert!(summary.passed());
}

#[test]
fn stage_pipeline_rejects_invalid_stage_counts() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut config = StagePipelineConfig::reference_decode();
    config.stages = 1;

    assert!(runtime.run_stage_pipeline_probe(config).is_err());
}
