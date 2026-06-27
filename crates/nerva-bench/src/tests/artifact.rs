use crate::artifact::run::run_artifact;
use crate::tests::support::{remove_tiny_hf_checkpoint_dir, write_tiny_hf_checkpoint_dir};

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
    assert!(json.contains("\"command_line\":[\"cargo\",\"run\",\"-p\",\"nerva-bench\""));
    assert!(json.contains("\"cwd\""));
    assert!(json.contains("\"git_commit\""));
    assert!(json.contains("\"package_version\""));
    assert!(json.contains("\"rustc_version\""));
    assert!(json.contains("\"cargo_version\""));
    assert!(json.contains("\"rustflags\""));
    assert!(json.contains("\"cargo_encoded_rustflags\""));
    assert!(json.contains("\"environment\""));
    assert!(json.contains("\"CUDA_VISIBLE_DEVICES\""));
    assert!(json.contains("\"HIP_VISIBLE_DEVICES\""));
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

#[test]
fn artifact_runs_external_hf_decode_checkpoint_with_metadata() {
    let dir = write_tiny_hf_checkpoint_dir("nerva-artifact-hf-decode");
    let json = run_artifact(
        Some("hf-decode".to_string()),
        vec![
            dir.to_string_lossy().into_owned(),
            "ids:0,1".to_string(),
            "2".to_string(),
        ],
    )
    .unwrap();

    assert!(json.contains("\"artifact_schema\":\"nerva-bench-v1\""));
    assert!(json.contains("\"command\":\"hf-decode\""));
    assert!(json.contains("\"args\":["));
    assert!(json.contains("\"capabilities\""));
    assert!(json.contains("\"summary\""));
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"input_mode\":\"token_ids\""));
    assert!(json.contains("\"context_mode\":\"prompt_prefill_kv_decode\""));
    assert!(json.contains("\"prompt_token_ids\":[0,1]"));
    assert!(json.contains("\"tokens\":[0,0]"));
    assert!(json.contains("\"manifest_entries\":12"));
    assert!(json.contains("\"shard_plan_entries\":12"));
    assert!(json.contains("\"hot_path_allocations\":0"));

    remove_tiny_hf_checkpoint_dir(&dir);
}

#[test]
fn artifact_runs_external_hf_cuda_decode_checkpoint_with_metadata() {
    let dir = write_tiny_hf_checkpoint_dir("nerva-artifact-hf-cuda-decode");
    let json = run_artifact(
        Some("hf-cuda-decode".to_string()),
        vec![
            dir.to_string_lossy().into_owned(),
            "ids:0,1".to_string(),
            "2".to_string(),
        ],
    )
    .unwrap();

    assert!(json.contains("\"artifact_schema\":\"nerva-bench-v1\""));
    assert!(json.contains("\"command\":\"hf-cuda-decode\""));
    assert!(json.contains("\"capabilities\""));
    assert!(json.contains("\"summary\""));
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"backend\":\"cuda\""));
    assert!(json.contains("\"input_mode\":\"token_ids\""));
    assert!(json.contains("\"prompt_token_ids\":[0,1]"));
    assert!(json.contains("\"resident_weight_plan\""));
    assert!(json.contains("\"cuda_contract_matched\":true"));
    assert!(json.contains("\"host_causality_edges\":0"));
    assert!(json.contains("\"hot_path_allocations\":0"));
    assert!(json.contains("\"critical_paths\":["));
    assert!(json.contains("\"proves_host_wait_not_gpu_idle\":true"));

    remove_tiny_hf_checkpoint_dir(&dir);
}
