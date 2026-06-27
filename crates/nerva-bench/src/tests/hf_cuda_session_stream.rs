use nerva_cuda::smoke::status::SmokeStatus;

use crate::cli::model::causal_lm_cuda_session_stream::hf_causal_lm_cuda_device_session_stream_json;
use crate::tests::support::{remove_tiny_hf_checkpoint_dir, write_tiny_hf_checkpoint_dir};

#[test]
fn hf_cuda_device_session_stream_cli_reports_bounded_records() {
    if nerva_runtime::capabilities::discovery::cuda_smoke().status != SmokeStatus::Ok {
        return;
    }
    let dir = write_tiny_hf_checkpoint_dir("nerva-hf-cuda-session-stream-cli");
    let json = hf_causal_lm_cuda_device_session_stream_json(
        Some(dir.to_string_lossy().into_owned()),
        3,
        1,
        2,
        1,
        Some("0".to_string()),
    )
    .unwrap();

    assert!(json.contains("\"mode\":\"device_session_stream\""));
    assert!(json.contains("\"queue\":{\"capacity\":1"));
    assert!(json.contains("\"pushes\":2"));
    assert!(json.contains("\"drains\":2"));
    assert!(json.contains("\"host_causality_edges\":0"));
    assert!(json.contains("\"device_authoritative\":true"));
    assert!(json.contains("\"host_causality_edge\":false"));
    assert!(json.contains("\"H2D_bytes\":0"));
    assert!(json.contains("\"graph_cache_hits\":1"));

    remove_tiny_hf_checkpoint_dir(&dir);
}
