use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::request::CUDA_HF_DECODE_SEQUENCE_DTYPE_F16;
use crate::decode::hf_sequence::session::request::CudaHfDecodeSequenceSessionConfig;
use crate::decode::hf_sequence::session::stateful::CudaHfDecodeSequenceLoop;
use crate::smoke::status::SmokeStatus;

#[test]
fn hf_decode_sequence_session_reuses_resident_weights_between_runs() {
    let _guard = super::cuda_test_lock();

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
        detailed_profile: false,
    }
    .create();

    if created.summary.status != SmokeStatus::Ok {
        return;
    }
    let mut session = created.session.unwrap();
    let first = session.run(&[0], 2, None);
    let second = session.run(&[1], 2, None);
    let third = session.run(&[0, 1], 1, None);

    assert!(session.create_summary().h2d_bytes > first.h2d_bytes);
    assert_eq!(first.tokens, vec![1, 2]);
    assert_eq!(second.tokens, vec![2, 3]);
    assert_eq!(third.tokens, vec![2]);
    assert_eq!(
        (first.h2d_bytes, second.h2d_bytes, third.h2d_bytes),
        (4, 4, 8)
    );
    assert_eq!((first.graph_nodes, first.kernel_launches), (3, 6));
    assert_eq!((first.graph_captures, first.graph_cache_hits), (1, 0));
    assert_eq!((second.graph_captures, second.graph_cache_hits), (0, 1));
    assert_eq!((third.graph_captures, third.graph_cache_hits), (1, 0));
    assert_eq!(first.host_causality_edges + second.host_causality_edges, 0);
    assert_eq!(
        first.hot_path_allocations + second.hot_path_allocations + third.hot_path_allocations,
        0
    );
    assert_eq!(
        first.device_free_memory_bytes,
        session.create_summary().device_free_memory_bytes
    );
    assert_eq!(
        first.fits_device_free_memory,
        session.create_summary().fits_device_free_memory
    );
    let create_json = session.create_summary().to_json();
    assert!(create_json.contains("\"fits_device_free_memory\":"));
    assert!(create_json.contains("\"H2D_bytes\":"));
    assert!(second.to_json().contains("\"graph_cache_hits\":1"));

    let started = CudaHfDecodeSequenceLoop::start(&mut session, &[0], None);
    assert_eq!(started.summary.status, SmokeStatus::Ok);
    assert_eq!(
        (started.summary.h2d_bytes, started.summary.d2h_bytes),
        (4, 0)
    );
    assert!(started.summary.kernel_launches > 0);
    assert!(started.summary.device_elapsed_ns > 0);
    let mut loop_state = started.loop_state.unwrap();
    let first_step = loop_state.advance(1);
    let second_step = loop_state.advance(1);
    assert_eq!(first_step.tokens, vec![1]);
    assert_eq!(second_step.tokens, vec![2]);
    assert_eq!((first_step.h2d_bytes, second_step.h2d_bytes), (0, 0));
    assert_eq!(
        (first_step.graph_captures, first_step.graph_cache_hits),
        (0, 0)
    );
    assert_eq!(first_step.kernel_launches, 0);
    assert_eq!(first_step.device_elapsed_ns, 0);
    assert_eq!(
        (second_step.graph_captures, second_step.graph_cache_hits),
        (0, 1)
    );
}

#[test]
fn hf_decode_sequence_session_packs_projection_replicas_for_cublas_path() {
    let _guard = super::cuda_test_lock();

    let one = 0x3c00;
    let zero = 0x0000;
    let hidden = 128;
    let intermediate = 256;
    let vocab_size = 8;
    let embeddings = vec![zero; vocab_size * hidden];
    let rms = vec![one; hidden];
    let attn_matrix = vec![zero; hidden * hidden];
    let mlp_matrix = vec![zero; intermediate * hidden];
    let down_matrix = vec![zero; hidden * intermediate];
    let lm_head = vec![zero; vocab_size * hidden];
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &attn_matrix,
        w_k: &attn_matrix,
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &attn_matrix,
        w_o: &attn_matrix,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &mlp_matrix,
        w_up: &mlp_matrix,
        w_down: &down_matrix,
    };
    let layers = [layer];
    let created = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden,
        heads: 1,
        kv_heads: 1,
        head_dim: hidden,
        intermediate,
        vocab_size,
        max_context_tokens: 1,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &embeddings,
        layers: &layers,
        final_norm_weight: &rms,
        lm_head: &lm_head,
        weight_plan: None,
        weight_blocks: &[],
        detailed_profile: false,
    }
    .create();

    if created.summary.status != SmokeStatus::Ok {
        return;
    }
    let mut session = created.session.unwrap();
    let started = CudaHfDecodeSequenceLoop::start(&mut session, &[0], None);
    assert_eq!(started.summary.status, SmokeStatus::Ok);
    assert!(started.summary.kernel_launches > 0);
    assert!(started.summary.device_elapsed_ns > 0);
    let mut loop_state = started.loop_state.unwrap();
    let summary = loop_state.advance(1);

    assert_eq!(summary.status, SmokeStatus::Ok);
    assert_eq!(summary.tokens, vec![0]);
    assert_eq!(summary.graph_replays, 0);
    assert_eq!(summary.graph_nodes, 0);
    assert_eq!(summary.kernel_launches, 0);
    assert_eq!(summary.device_elapsed_ns, 0);
    assert!(summary.device_arena_bytes > summary.resident_weight_bytes);
    assert_eq!(summary.host_causality_edges, 0);
    assert_eq!(summary.hot_path_allocations, 0);
}
