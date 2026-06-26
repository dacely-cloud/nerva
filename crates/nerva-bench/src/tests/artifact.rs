use crate::artifact::run::run_artifact;

#[test]
fn artifact_wraps_probe_with_reproducibility_metadata() {
    let json = run_artifact(
        Some("synthetic".to_string()),
        vec!["2".to_string(), "4".to_string()],
    )
    .unwrap();

    assert!(json.contains("\"artifact_schema\":\"nerva-bench-v1\""));
    assert!(json.contains("\"command\":\"synthetic\""));
    assert!(json.contains("\"args\":[\"2\",\"4\"]"));
    assert!(json.contains("\"git_commit\""));
    assert!(json.contains("\"package_version\""));
    assert!(json.contains("\"rustc_version\""));
    assert!(json.contains("\"cargo_version\""));
    assert!(json.contains("\"rustflags\""));
    assert!(json.contains("\"cargo_encoded_rustflags\""));
    assert!(json.contains("\"capabilities\""));
    assert!(json.contains("\"target_os\":\"linux\""));
    assert!(json.contains("\"cuda_compute_capability\""));
    assert!(json.contains("\"cuda_device_total_memory_bytes\""));
    assert!(json.contains("\"cuda_pci_bus_id\""));
    assert!(json.contains("\"rdma_core_loaded\""));
    assert!(json.contains("\"mlx5_core_loaded\""));
    assert!(json.contains("\"nvidia_peer_memory_module\""));
    assert!(json.contains("\"topology\""));
    assert!(json.contains("\"summary\""));
    assert!(json.contains("\"observed_token_hash\""));
    assert!(json.contains("\"token_ring_reuses\""));
    assert!(json.contains("\"device_timeline_idle_ns\":0"));
}
