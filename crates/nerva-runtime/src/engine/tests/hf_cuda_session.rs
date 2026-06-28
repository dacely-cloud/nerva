use nerva_core::types::id::token::TokenId;
use nerva_cuda::smoke::status::SmokeStatus;

use crate::engine::hf_cuda_decode::file_backed::session::create_hf_causal_lm_cuda_shard_backed_device_only_session;
use crate::engine::runtime::{Runtime, RuntimeConfig};
use crate::engine::tests::hf_fixture::{remove_hf_checkpoint_dir, write_kv_hf_checkpoint_dir};

#[test]
fn file_backed_hf_cuda_session_reuses_weight_uploads_across_runs() {
    let _guard = super::cuda_test_lock();

    if crate::capabilities::discovery::cuda_smoke().status != SmokeStatus::Ok {
        return;
    }
    let dir = write_kv_hf_checkpoint_dir("nerva-hf-cuda-session");
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let mut session =
        create_hf_causal_lm_cuda_shard_backed_device_only_session(&runtime, &dir, 4, Some(120))
            .unwrap();
    remove_hf_checkpoint_dir(&dir);

    let first = session.run(&[TokenId(0)], 2).unwrap();
    let second = session.run(&[TokenId(1)], 2).unwrap();

    assert_eq!(first.status, SmokeStatus::Ok);
    assert_eq!(second.status, SmokeStatus::Ok);
    assert_eq!((first.h2d_bytes, second.h2d_bytes), (4, 4));
    assert!(session.create_summary.h2d_bytes > first.h2d_bytes);
    assert!(session.create_summary.descriptor_gpu_staged_h2d_bytes > 0);
    assert_eq!(first.resident_weights.cuda_contract_matched, true);
    assert_eq!(first.host_causality_edges + second.host_causality_edges, 0);
    assert_eq!(first.hot_path_allocations + second.hot_path_allocations, 0);
    assert_eq!((first.graph_nodes, first.kernel_launches), (3, 6));
    assert_eq!((first.graph_captures, first.graph_cache_hits), (1, 0));
    assert_eq!((second.graph_captures, second.graph_cache_hits), (0, 1));
}
