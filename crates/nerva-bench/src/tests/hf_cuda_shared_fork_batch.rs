use crate::cli::model::causal_lm_cuda_shared_fork_batch::{
    hf_causal_lm_cuda_shared_fork_batch_compare_json, hf_causal_lm_cuda_shared_fork_batch_json,
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
    )
    .unwrap_err();

    assert_eq!(
        err,
        "hf-cuda-shared-fork-batch-compare requires checkpoint_dir"
    );
}
