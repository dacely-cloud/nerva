use crate::backend::probe::backend_contract_smoke;
use crate::backend::summary::CudaBackendContractSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn backend_contract_summary_serializes_allocation_and_queue_fields() {
    let summary = CudaBackendContractSummary {
        status: SmokeStatus::Ok,
        gpu_name: Some("RTX".to_string()),
        driver_version: Some(13_010),
        runtime_version: Some(13_010),
        compute_capability_major: Some(12),
        compute_capability_minor: Some(0),
        device_total_memory_bytes: Some(32 * 1024 * 1024 * 1024),
        device_free_memory_bytes: Some(31 * 1024 * 1024 * 1024),
        pci_bus_id: Some("0000:01:00.0".to_string()),
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
        d2h_bytes: core::mem::size_of::<u32>() as u64,
        sync_calls: 1,
        observed_word: Some(0x5a5a_5a5a),
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(summary.passed());
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"requested_device_bytes\":4096"));
    assert!(json.contains("\"device_free_memory_bytes\":33285996544"));
    assert!(json.contains("\"allocated_device_bytes\":4096"));
    assert!(json.contains("\"requested_pinned_bytes\":4096"));
    assert!(json.contains("\"allocated_pinned_bytes\":4096"));
    assert!(json.contains("\"stream_creations\":1"));
    assert!(json.contains("\"stream_destroys\":1"));
    assert!(json.contains("\"event_creations\":1"));
    assert!(json.contains("\"event_destroys\":1"));
    assert!(json.contains("\"device_allocations\":1"));
    assert!(json.contains("\"device_frees\":1"));
    assert!(json.contains("\"pinned_allocations\":1"));
    assert!(json.contains("\"pinned_frees\":1"));
    assert!(json.contains("\"memset_bytes\":4096"));
    assert!(json.contains("\"D2H_bytes\":4"));
    assert!(json.contains("\"sync_calls\":1"));
    assert!(json.contains("\"observed_word\":1515870810"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn backend_contract_smoke_is_repeatable_when_device_is_available() {
    let first = backend_contract_smoke(4096, 4096);
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = backend_contract_smoke(4096, 4096);
    assert!(first.passed(), "first backend contract: {first:?}");
    assert!(second.passed(), "second backend contract: {second:?}");
    assert_eq!(second.observed_word, Some(0x5a5a_5a5a));
    assert_eq!(second.hot_path_allocations, 0);
}
