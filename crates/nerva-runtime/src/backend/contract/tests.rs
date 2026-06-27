use nerva_core::types::backend::contract::DeviceBackend;
use nerva_core::types::backend::operation::BackendTransactionDescriptor;
use nerva_core::types::dtype::DType;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_core::types::id::transaction::TransactionId;

use nerva_cuda::backend::summary::CudaBackendContractSummary;
use nerva_cuda::graph::summary::CudaSyntheticGraphSummary;
use nerva_cuda::sampler::summary::CudaGreedySamplerSummary;
use nerva_cuda::smoke::status::SmokeStatus;

use crate::backend::contract::adapter::CudaBackendContractAdapter;
use crate::backend::contract::status::BackendContractProbeStatus;
use crate::engine::runtime::{Runtime, RuntimeConfig};

#[test]
fn backend_contract_probe_reports_neutral_cuda_surface_when_available() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_backend_contract_probe();
    if summary.status != BackendContractProbeStatus::Ok {
        return;
    }

    assert!(summary.passed(), "{summary:?}");
    assert_eq!(summary.backend.as_str(), "cuda");
    assert!(summary.supports_device_allocations);
    assert!(summary.supports_pinned_host_allocations);
    assert!(summary.supports_streams);
    assert!(summary.supports_events);
    assert!(summary.supports_graph_capture);
    assert!(summary.supports_async_copies);
    assert!(summary.supports_device_sampling);
    assert!(summary.exact_dtypes.contains(&DType::F16));
    assert!(!summary.exact_dtypes.contains(&DType::BF16));
    assert_eq!(summary.hot_path_allocations, 0);
}

#[test]
fn cuda_backend_adapter_rejects_unproven_allocation_bytes() {
    let backend = successful_backend_summary();
    let graph = successful_graph_summary();
    let sampler = successful_sampler_summary();
    let adapter = CudaBackendContractAdapter::from_probe(&backend, &graph, &sampler).unwrap();
    let device = adapter.create_device(DeviceOrdinal(0)).unwrap();

    assert!(adapter.allocate_device(&device, 4096, 256).is_ok());
    assert!(adapter.allocate_device(&device, 8192, 256).is_err());
    assert!(adapter.allocate_pinned(4096, 64).is_ok());
    assert!(adapter.allocate_pinned(8192, 64).is_err());
}

#[test]
fn cuda_backend_adapter_rejects_non_capturable_transactions() {
    let backend = successful_backend_summary();
    let graph = successful_graph_summary();
    let sampler = successful_sampler_summary();
    let adapter = CudaBackendContractAdapter::from_probe(&backend, &graph, &sampler).unwrap();
    let transaction = BackendTransactionDescriptor {
        id: TransactionId(9),
        operation_count: 1,
        block_use_count: 1,
        graph_capturable: false,
    };

    assert!(adapter.capture(&transaction).is_err());
}

fn successful_backend_summary() -> CudaBackendContractSummary {
    CudaBackendContractSummary {
        status: SmokeStatus::Ok,
        gpu_name: Some("CUDA device".to_string()),
        driver_version: Some(13_010),
        runtime_version: Some(13_010),
        compute_capability_major: Some(12),
        compute_capability_minor: Some(0),
        device_total_memory_bytes: Some(32 * 1024 * 1024 * 1024),
        device_free_memory_bytes: Some(31 * 1024 * 1024 * 1024),
        pci_bus_id: Some("0000:65:00.0".to_string()),
        device_count: 1,
        device_ordinal: 0,
        requested_device_bytes: 4096,
        requested_pinned_bytes: 4096,
        allocated_device_bytes: 4096,
        allocated_pinned_bytes: 4096,
        stream_creations: 1,
        stream_destroys: 1,
        event_creations: 1,
        event_destroys: 1,
        device_allocations: 1,
        device_frees: 1,
        pinned_allocations: 1,
        pinned_frees: 1,
        memset_bytes: 4096,
        d2h_bytes: 4,
        sync_calls: 1,
        observed_word: Some(0x5a5a_5a5a),
        hot_path_allocations: 0,
        error: None,
    }
}

fn successful_graph_summary() -> CudaSyntheticGraphSummary {
    CudaSyntheticGraphSummary {
        status: SmokeStatus::Ok,
        steps: 16,
        ring_capacity: 4,
        seed_token: 1,
        last_token: Some(1),
        graph_replays: 16,
        graph_nodes: 2,
        observed_tokens: 16,
        observed_token_hash: 1,
        token_ring_slots_touched: 4,
        token_ring_reuses: 12,
        token_ring_max_slot_version: 4,
        stale_tokens: 0,
        missing_tokens: 0,
        extra_tokens: 0,
        mismatched_tokens: 0,
        host_causality_edges: 0,
        device_arena_bytes: 1024,
        pinned_host_bytes: 128,
        graph_launches: 16,
        sync_calls: 16,
        d2h_bytes: 64,
        hot_path_allocations: 0,
        error: None,
    }
}

fn successful_sampler_summary() -> CudaGreedySamplerSummary {
    CudaGreedySamplerSummary {
        status: SmokeStatus::Ok,
        vocab_size: 4,
        token_index: 0,
        token: 2,
        slot_version: 1,
        completion: 1,
        device_arena_bytes: 56,
        pinned_host_bytes: 56,
        h2d_bytes: 16,
        d2h_bytes: 40,
        kernel_launches: 1,
        sync_calls: 2,
        hot_path_allocations: 0,
        error: None,
    }
}
