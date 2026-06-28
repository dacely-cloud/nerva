use nerva_cuda::smoke::status::SmokeStatus;

use crate::cli::model::causal_lm_cuda_generate::hf_causal_lm_cuda_generate_json;
use crate::tests::support::{
    remove_tiny_hf_checkpoint_dir, write_tiny_hf_checkpoint_dir, write_tiny_wordlevel_tokenizer,
};

#[test]
fn hf_cuda_generate_cli_reports_user_facing_generation() {
    if nerva_runtime::capabilities::discovery::cuda_smoke().status != SmokeStatus::Ok {
        return;
    }
    let dir = write_tiny_hf_checkpoint_dir("nerva-hf-cuda-generate-cli");
    write_tiny_wordlevel_tokenizer(&dir);
    let json = hf_causal_lm_cuda_generate_json(
        Some(dir.to_string_lossy().into_owned()),
        4,
        2,
        1,
        Some("one two".to_string()),
        None,
    )
    .unwrap();

    assert!(json.contains("\"mode\":\"device_generate\""));
    assert!(json.contains("\"input_mode\":\"tokenizer_json\""));
    assert!(json.contains("\"prompt\":\"one two\""));
    assert!(json.contains("\"prompt_token_ids\":[1,2]"));
    assert!(json.contains("\"max_new_tokens\":2"));
    assert!(json.contains("\"stop_reason\":\"max_steps\""));
    assert!(json.contains("\"chunks_observed\":2"));
    assert!(json.contains("\"perf\":{\"generated_tokens\":2"));
    assert!(json.contains("\"tokens_per_second\":"));
    assert!(json.contains("\"token_p99_ms\":"));
    assert!(json.contains("\"graph_nodes_per_token\":"));
    assert!(json.contains("\"timing_source\":\"replay_critical_path_gpu_events\""));
    assert!(json.contains("\"profile_bucket_source\":\"profile_pass_gpu_events\""));
    assert!(json.contains("\"measured_replay_ns_per_token\":"));
    assert!(json.contains("\"profile_accounted_ns_per_token\":"));
    assert!(json.contains("\"profile_replay_ratio\":"));
    assert!(json.contains("\"profile_projection_ns_per_token\":"));
    assert!(json.contains("\"queue\":{\"capacity\":1"));
    assert!(json.contains("\"host_causality_edges\":0"));
    assert!(json.contains("\"device_authoritative\":true"));
    assert!(json.contains("\"H2D_bytes\":0"));
    assert!(json.contains("\"graph_cache_hits\":1"));

    remove_tiny_hf_checkpoint_dir(&dir);
}

#[test]
fn hf_cuda_generate_cli_requires_checkpoint_dir() {
    let err =
        hf_causal_lm_cuda_generate_json(None, 3, 2, 1, Some("one".to_string()), None).unwrap_err();

    assert_eq!(err, "hf-cuda-generate requires checkpoint_dir");
}
