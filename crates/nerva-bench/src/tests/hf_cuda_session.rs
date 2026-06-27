use nerva_cuda::smoke::status::SmokeStatus;

use crate::cli::model::causal_lm_cuda_session::hf_causal_lm_cuda_device_session_json;
use crate::tests::support::{remove_tiny_hf_checkpoint_dir, write_tiny_hf_checkpoint_dir};

#[test]
fn hf_cuda_device_session_cli_reports_create_once_and_multiple_runs() {
    if nerva_runtime::capabilities::discovery::cuda_smoke().status != SmokeStatus::Ok {
        return;
    }
    let dir = write_tiny_hf_checkpoint_dir("nerva-hf-cuda-session-cli");
    let json = hf_causal_lm_cuda_device_session_json(
        Some(dir.to_string_lossy().into_owned()),
        3,
        2,
        vec!["0".to_string(), "1".to_string()],
    )
    .unwrap();

    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"mode\":\"device_session\""));
    assert!(json.contains("\"create\":{"));
    assert!(json.contains("\"runs\":["));
    assert!(json.contains("\"input\":\"0\""));
    assert!(json.contains("\"input\":\"1\""));
    assert!(json.contains("\"reference_mode\":\"device_only_unverified\""));
    assert!(json.contains("\"H2D_bytes\":4"));
    assert!(json.contains("\"host_causality_edges\":0"));
    assert!(json.contains("\"hot_path_allocations\":0"));

    remove_tiny_hf_checkpoint_dir(&dir);
}
