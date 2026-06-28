use nerva_core::types::id::token::TokenId;
use nerva_cuda::smoke::status::SmokeStatus;

use crate::engine::hf_cuda_decode::file_backed::session_loop::run_hf_causal_lm_cuda_shard_backed_device_session_loop;
use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::engine::tests::hf_fixture::{remove_hf_checkpoint_dir, write_kv_hf_checkpoint_dir};

#[test]
fn file_backed_hf_cuda_session_loop_advances_without_prompt_h2d() {
    let _guard = super::cuda_test_lock();

    if crate::capabilities::discovery::cuda_smoke().status != SmokeStatus::Ok {
        return;
    }
    let dir = write_kv_hf_checkpoint_dir("nerva-hf-cuda-session-loop");
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let output = run_hf_causal_lm_cuda_shard_backed_device_session_loop(
        &runtime,
        &dir,
        &[TokenId(0)],
        3,
        1,
        2,
        Some(120),
    )
    .unwrap();
    remove_hf_checkpoint_dir(&dir);

    assert_eq!(output.start.status, SmokeStatus::Ok);
    assert_eq!((output.start.h2d_bytes, output.start.d2h_bytes), (4, 0));
    assert!(output.start.kernel_launches > 0);
    assert!(output.start.device_elapsed_ns > 0);
    assert_eq!(output.tokens.len(), 2);
    assert_eq!(output.tokens[0], TokenId(1));
    assert_eq!(output.chunks.len(), 2);
    let first = &output.chunks[0].summary;
    let second = &output.chunks[1].summary;
    assert_eq!((first.h2d_bytes, second.h2d_bytes), (0, 0));
    assert_eq!(first.host_causality_edges + second.host_causality_edges, 0);
    assert_eq!(first.hot_path_allocations + second.hot_path_allocations, 0);
}
