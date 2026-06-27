use nerva_core::types::id::token::TokenId;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::causal_lm::types::HfCausalLmStopReason;

use crate::engine::hf_cuda_decode::file_backed::generate::run_hf_causal_lm_cuda_shard_backed_device_generate;
use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::engine::tests::hf_fixture::{remove_hf_checkpoint_dir, write_kv_hf_checkpoint_dir};

#[test]
fn file_backed_hf_cuda_generate_uses_stateful_stream_path() {
    if crate::capabilities::discovery::cuda_smoke().status != SmokeStatus::Ok {
        return;
    }
    let dir = write_kv_hf_checkpoint_dir("nerva-hf-cuda-generate");
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let output = run_hf_causal_lm_cuda_shard_backed_device_generate(
        &runtime,
        &dir,
        &[TokenId(0)],
        3,
        2,
        2,
        Some(120),
    )
    .unwrap();
    remove_hf_checkpoint_dir(&dir);

    assert_eq!(output.max_new_tokens, 2);
    assert_eq!(output.stop_reason(), HfCausalLmStopReason::MaxSteps);
    assert_eq!(output.tokens().len(), 2);
    assert_eq!(
        (output.stream.start.h2d_bytes, output.stream.start.d2h_bytes),
        (4, 0)
    );
    assert_eq!(output.stream.queue.host_causality_edges, 0);
    assert_eq!(
        output.stream.chunks[0].h2d_bytes + output.stream.chunks[1].h2d_bytes,
        0
    );
    assert_eq!(output.stream.chunks[1].graph_cache_hits, 1);
    assert!(
        output
            .stream
            .records
            .iter()
            .all(|record| record.device_authoritative && !record.host_causality_edge)
    );
}
