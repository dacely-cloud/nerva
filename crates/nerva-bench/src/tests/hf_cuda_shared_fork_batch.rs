use crate::cli::model::causal_lm_cuda_shared_fork_batch::{
    SHARED_FORK_STORY_PROMPT, hf_causal_lm_cuda_shared_fork_batch_compare_json,
    hf_causal_lm_cuda_shared_fork_batch_json, strip_experimental_rt_arg,
};

#[test]
fn hf_cuda_shared_fork_batch_cli_requires_checkpoint_dir() {
    let err = hf_causal_lm_cuda_shared_fork_batch_json(
        None,
        2,
        8,
        2,
        2,
        2,
        Some("one".to_string()),
        None,
        false,
    )
    .unwrap_err();

    assert_eq!(err, "hf-cuda-shared-fork-batch requires checkpoint_dir");
}

#[test]
fn hf_cuda_shared_fork_batch_compare_cli_requires_checkpoint_dir() {
    let err = hf_causal_lm_cuda_shared_fork_batch_compare_json(
        None,
        2,
        8,
        2,
        2,
        2,
        Some("one".to_string()),
        None,
        false,
    )
    .unwrap_err();

    assert_eq!(
        err,
        "hf-cuda-shared-fork-batch-compare requires checkpoint_dir"
    );
}

#[test]
fn hf_cuda_shared_fork_batch_default_prompt_is_real_generation_workload() {
    let lower = SHARED_FORK_STORY_PROMPT.to_ascii_lowercase();

    assert!(SHARED_FORK_STORY_PROMPT.len() > 100);
    assert!(lower.contains("story"));
    assert!(lower.contains("detail"));
    assert_ne!(lower.trim(), "hello");
}

#[test]
fn hf_cuda_shared_fork_batch_strips_experimental_rt_flag() {
    let args = vec![
        "--experimental-rt".to_string(),
        "checkpoint".to_string(),
        "4".to_string(),
        "--experimental-rt".to_string(),
    ];
    let (items, enabled) = strip_experimental_rt_arg(&mut args.into_iter());

    assert!(enabled);
    assert_eq!(items, ["checkpoint", "4"]);
}
