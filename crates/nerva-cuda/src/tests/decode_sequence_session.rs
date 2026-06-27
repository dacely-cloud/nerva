use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::request::CUDA_HF_DECODE_SEQUENCE_DTYPE_F16;
use crate::decode::hf_sequence::session::request::CudaHfDecodeSequenceSessionConfig;
use crate::smoke::status::SmokeStatus;

#[test]
fn hf_decode_sequence_session_reuses_resident_weights_between_runs() {
    let one = 0x3c00;
    let zero = 0x0000;
    let neg_one = 0xbc00;
    let embeddings = [one, zero, zero, one, neg_one, zero, zero, neg_one];
    let rms = [one, one];
    let matrix = [zero; 4];
    let lm_head = [zero, neg_one, one, zero, zero, one, neg_one, zero];
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &matrix,
        w_k: &matrix,
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &matrix,
        w_o: &matrix,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &matrix,
        w_up: &matrix,
        w_down: &matrix,
    };
    let layers = [layer];
    let created = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        vocab_size: 4,
        max_context_tokens: 2,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &embeddings,
        layers: &layers,
        final_norm_weight: &rms,
        lm_head: &lm_head,
        weight_plan: None,
        weight_blocks: &[],
    }
    .create();

    if created.summary.status != SmokeStatus::Ok {
        return;
    }
    let mut session = created.session.unwrap();
    let first = session.run(&[0], 2, None);
    let second = session.run(&[1], 2, None);

    assert!(session.create_summary().h2d_bytes > first.h2d_bytes);
    assert_eq!(first.tokens, vec![1, 2]);
    assert_eq!(second.tokens, vec![2, 3]);
    assert_eq!((first.h2d_bytes, second.h2d_bytes), (4, 4));
    assert_eq!((first.graph_nodes, first.kernel_launches), (3, 6));
    assert_eq!(first.host_causality_edges + second.host_causality_edges, 0);
    assert_eq!(first.hot_path_allocations + second.hot_path_allocations, 0);
    let create_json = session.create_summary().to_json();
    assert!(create_json.contains("\"H2D_bytes\":"));
}
