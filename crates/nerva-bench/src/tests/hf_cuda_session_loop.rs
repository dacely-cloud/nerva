use nerva_cuda::smoke::status::SmokeStatus;

use crate::cli::model::causal_lm_cuda_session_loop::hf_causal_lm_cuda_device_session_loop_json;
use crate::tests::support::{remove_tiny_hf_checkpoint_dir, write_tiny_hf_checkpoint_dir};

#[test]
fn hf_cuda_device_session_loop_cli_reports_stateful_chunks() {
    if nerva_runtime::capabilities::discovery::cuda_smoke().status != SmokeStatus::Ok {
        return;
    }
    let dir = write_tiny_hf_checkpoint_dir("nerva-hf-cuda-session-loop-cli");
    let json = hf_causal_lm_cuda_device_session_loop_json(
        Some(dir.to_string_lossy().into_owned()),
        3,
        1,
        2,
        Some("0".to_string()),
    )
    .unwrap();

    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"mode\":\"device_session_loop\""));
    assert!(json.contains("\"start\":{"));
    assert!(json.contains("\"chunks_observed\":2"));
    assert!(json.contains("\"H2D_bytes\":0"));
    assert!(json.contains("\"graph_captures\":1"));
    assert!(json.contains("\"graph_cache_hits\":1"));
    assert!(json.contains("\"host_causality_edges\":0"));
    assert!(json.contains("\"hot_path_allocations\":0"));

    remove_tiny_hf_checkpoint_dir(&dir);
}
