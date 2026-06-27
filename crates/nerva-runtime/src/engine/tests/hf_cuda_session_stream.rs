use nerva_core::types::id::token::TokenId;
use nerva_cuda::smoke::status::SmokeStatus;

use crate::engine::hf_cuda_decode::file_backed::session_stream::run_hf_causal_lm_cuda_shard_backed_device_session_stream;
use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::engine::tests::hf_fixture::{remove_hf_checkpoint_dir, write_kv_hf_checkpoint_dir};

#[test]
fn file_backed_hf_cuda_session_stream_uses_bounded_host_queue() {
    if crate::capabilities::discovery::cuda_smoke().status != SmokeStatus::Ok {
        return;
    }
    let dir = write_kv_hf_checkpoint_dir("nerva-hf-cuda-session-stream");
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let output = run_hf_causal_lm_cuda_shard_backed_device_session_stream(
        &runtime,
        &dir,
        &[TokenId(0)],
        3,
        1,
        2,
        1,
        Some(120),
    )
    .unwrap();
    remove_hf_checkpoint_dir(&dir);

    assert_eq!(output.start.status, SmokeStatus::Ok);
    assert_eq!((output.start.h2d_bytes, output.start.d2h_bytes), (4, 0));
    assert_eq!(output.records.len(), 2);
    assert_eq!((output.queue.capacity, output.queue.high_watermark), (1, 1));
    assert_eq!((output.queue.pushes, output.queue.drains), (2, 2));
    assert_eq!(output.queue.overflow_rejections, 0);
    assert_eq!(output.queue.host_causality_edges, 0);
    assert!(
        output
            .records
            .iter()
            .all(|record| record.device_authoritative)
    );
    assert!(
        output
            .records
            .iter()
            .all(|record| !record.host_causality_edge)
    );
    assert_eq!(output.chunks[0].h2d_bytes + output.chunks[1].h2d_bytes, 0);
    assert_eq!(output.chunks[1].graph_cache_hits, 1);
}
