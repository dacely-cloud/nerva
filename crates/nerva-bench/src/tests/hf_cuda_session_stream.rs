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
    assert!(json.contains("\"stop_reason\":\"max_steps\""));
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

#[test]
fn hf_cuda_device_session_stream_cli_reports_eos_stop() {
    if nerva_runtime::capabilities::discovery::cuda_smoke().status != SmokeStatus::Ok {
        return;
    }
    let dir = write_tiny_hf_checkpoint_dir("nerva-hf-cuda-session-stream-eos-cli");
    write_config_with_eos_zero(&dir);
    let json = hf_causal_lm_cuda_device_session_stream_json(
        Some(dir.to_string_lossy().into_owned()),
        4,
        2,
        3,
        2,
        Some("0".to_string()),
    )
    .unwrap();

    assert!(json.contains("\"stop_reason\":\"eos_token\""));
    assert!(json.contains("\"chunks_observed\":1"));
    assert!(json.contains("\"pushes\":1"));
    assert!(json.contains("\"host_causality_edges\":0"));
    assert!(json.contains("\"tokens\":[0]"));

    remove_tiny_hf_checkpoint_dir(&dir);
}

fn write_config_with_eos_zero(dir: &std::path::Path) {
    let config = r#"{
            "model_type": "llama",
            "hidden_size": 2,
            "intermediate_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 4,
            "eos_token_id": 0,
            "torch_dtype": "float16"
        }"#;
    std::fs::write(dir.join("config.json"), config).unwrap();
}
