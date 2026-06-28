use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::request::CUDA_HF_DECODE_SEQUENCE_DTYPE_F16;
use crate::decode::hf_sequence::session::request::{
    CudaHfDecodeSequenceLayerProjectionBatchExecuteSummary,
    CudaHfDecodeSequenceProjectionBatchExecuteSummary, CudaHfDecodeSequenceSession,
    CudaHfDecodeSequenceSessionConfig,
};
use crate::decode::hf_sequence::session::stateful::CudaHfDecodeSequenceLoop;
use crate::decode::hf_sequence::weight_plan::{
    hash_weight_blocks, CudaHfDecodeSequenceWeightBlock, CudaHfDecodeSequenceWeightPlan,
    CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
};
use crate::smoke::status::SmokeStatus;

#[test]
fn hf_decode_sequence_projection_batch_plan_reports_no_sessions_without_cuda() {
    let summary = CudaHfDecodeSequenceSession::projection_batch_plan(&mut [], 8, 2);

    assert_eq!(summary.status, SmokeStatus::Ok);
    assert_eq!(summary.reason, "no_sessions");
    assert!(!summary.exact);
    assert_eq!(summary.requested_session_count, 0);
    assert_eq!(summary.block_tokens, 0);
    assert_eq!(summary.target_block_tokens, 8);
    assert_eq!(summary.min_block_tokens, 2);
    assert_eq!(summary.hot_path_allocations, 0);
}

#[test]
fn hf_decode_sequence_projection_batch_execute_reports_no_sessions_without_cuda() {
    let summary = CudaHfDecodeSequenceSession::execute_qkv_projection_batch(&mut [], 8, 2, 0);

    assert_eq!(summary.status, SmokeStatus::Ok);
    assert_eq!(summary.reason, "no_sessions");
    assert!(!summary.exact);
    assert_eq!(summary.requested_session_count, 0);
    assert_eq!(summary.block_tokens, 0);
    assert_eq!(summary.projection_kind, 1);
    assert_eq!(summary.layer_index, 0);
    assert_eq!(summary.hot_path_allocations, 0);
}

#[test]
fn hf_decode_sequence_layer_projection_batch_execute_reports_no_sessions_without_cuda() {
    let summary = CudaHfDecodeSequenceSession::execute_layer_projection_batch(&mut [], 8, 2, 0);

    assert_eq!(summary.status, SmokeStatus::Ok);
    assert_eq!(summary.reason, "no_sessions");
    assert!(!summary.exact);
    assert_eq!(summary.requested_session_count, 0);
    assert_eq!(summary.block_tokens, 0);
    assert_eq!(summary.layer_index, 0);
    assert_eq!(summary.hot_path_allocations, 0);
}

#[test]
fn hf_decode_sequence_projection_batch_executes_all_projection_kinds_for_two_sessions() {
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
    let weight_blocks = projection_batch_weight_blocks(
        &embeddings,
        &rms,
        &attn_matrix,
        &mlp_matrix,
        &down_matrix,
        &lm_head,
    );
    let weight_plan = projection_batch_weight_plan(&weight_blocks);
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden,
        heads: 1,
        kv_heads: 1,
        head_dim: hidden,
        intermediate,
        vocab_size,
        max_context_tokens: 4,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &embeddings,
        layers: &layers,
        final_norm_weight: &rms,
        lm_head: &lm_head,
        weight_plan: Some(weight_plan),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
    };
    let created_a = config.create();
    if created_a.summary.status != SmokeStatus::Ok {
        return;
    }
    let created_b = config.create();
    if created_b.summary.status != SmokeStatus::Ok {
        return;
    }
    let mut session_a = created_a.session.unwrap();
    let mut session_b = created_b.session.unwrap();

    {
        let started = CudaHfDecodeSequenceLoop::start(&mut session_a, &[0], None);
        if started.summary.status != SmokeStatus::Ok {
            return;
        }
    }
    {
        let started = CudaHfDecodeSequenceLoop::start(&mut session_b, &[1], None);
        if started.summary.status != SmokeStatus::Ok {
            return;
        }
    }

    let plan = {
        let mut sessions = [&mut session_a, &mut session_b];
        CudaHfDecodeSequenceSession::projection_batch_plan(&mut sessions, 2, 2)
    };
    assert_eq!(plan.status, SmokeStatus::Ok);
    assert_eq!(plan.reason, "ready");
    assert!(plan.exact);
    assert_eq!(plan.block_tokens, 2);
    assert_eq!(plan.qkv_rows, (hidden * 3) as u64);
    assert_eq!(plan.pack_input_bytes, (intermediate * 2 * 2) as u64);
    assert_eq!(plan.hot_path_allocations, 0);

    let qkv = {
        let mut sessions = [&mut session_a, &mut session_b];
        CudaHfDecodeSequenceSession::execute_qkv_projection_batch(&mut sessions, 2, 2, 0)
    };
    assert_projection_batch_exec(&qkv, 1, (hidden * 3) as u32, hidden as u32);

    let attention_output = {
        let mut sessions = [&mut session_a, &mut session_b];
        CudaHfDecodeSequenceSession::execute_attention_output_projection_batch(
            &mut sessions,
            2,
            2,
            0,
        )
    };
    assert_projection_batch_exec(&attention_output, 2, hidden as u32, hidden as u32);

    let gate_up = {
        let mut sessions = [&mut session_a, &mut session_b];
        CudaHfDecodeSequenceSession::execute_gate_up_projection_batch(&mut sessions, 2, 2, 0)
    };
    assert_projection_batch_exec(&gate_up, 3, (intermediate * 2) as u32, hidden as u32);

    let down = {
        let mut sessions = [&mut session_a, &mut session_b];
        CudaHfDecodeSequenceSession::execute_down_projection_batch(&mut sessions, 2, 2, 0)
    };
    assert_projection_batch_exec(&down, 4, hidden as u32, intermediate as u32);

    let lm_head = {
        let mut sessions = [&mut session_a, &mut session_b];
        CudaHfDecodeSequenceSession::execute_lm_head_projection_batch(&mut sessions, 2, 2)
    };
    assert_projection_batch_exec(&lm_head, 5, vocab_size as u32, hidden as u32);

    let layer = {
        let mut sessions = [&mut session_a, &mut session_b];
        CudaHfDecodeSequenceSession::execute_layer_projection_batch(&mut sessions, 2, 2, 0)
    };
    assert_layer_projection_batch_exec(
        &layer,
        (hidden * 3) as u32,
        hidden as u32,
        (intermediate * 2) as u32,
        hidden as u32,
        hidden as u32,
        intermediate as u32,
    );
}

fn assert_projection_batch_exec(
    summary: &CudaHfDecodeSequenceProjectionBatchExecuteSummary,
    projection_kind: u32,
    rows: u32,
    cols: u32,
) {
    assert_eq!(summary.status, SmokeStatus::Ok);
    assert_eq!(summary.reason, "ready");
    assert!(summary.exact);
    assert_eq!(summary.projection_kind, projection_kind);
    assert_eq!(summary.block_tokens, 2);
    assert_eq!(summary.rows, rows);
    assert_eq!(summary.cols, cols);
    assert_eq!(summary.input_bytes, u64::from(cols) * 2 * 2);
    assert_eq!(summary.output_bytes, u64::from(rows) * 2 * 4);
    assert_eq!(summary.pack_kernel_launches, 2);
    assert_eq!(summary.projection_kernel_launches, 1);
    assert_eq!(summary.scatter_kernel_launches, 2);
    assert!(summary.elapsed_ns > 0);
    assert_eq!(summary.hot_path_allocations, 0);
}

fn assert_layer_projection_batch_exec(
    summary: &CudaHfDecodeSequenceLayerProjectionBatchExecuteSummary,
    qkv_rows: u32,
    attention_output_rows: u32,
    gate_up_rows: u32,
    down_rows: u32,
    hidden_cols: u32,
    down_cols: u32,
) {
    assert_eq!(summary.status, SmokeStatus::Ok);
    assert_eq!(summary.reason, "ready");
    assert!(summary.exact);
    assert_eq!(summary.layer_index, 0);
    assert_eq!(summary.block_tokens, 2);
    assert_eq!(summary.qkv_rows, qkv_rows);
    assert_eq!(summary.attention_output_rows, attention_output_rows);
    assert_eq!(summary.gate_up_rows, gate_up_rows);
    assert_eq!(summary.down_rows, down_rows);
    assert_eq!(summary.hidden_cols, hidden_cols);
    assert_eq!(summary.attention_output_cols, hidden_cols);
    assert_eq!(summary.down_cols, down_cols);
    assert_eq!(
        summary.input_bytes,
        u64::from(hidden_cols + summary.attention_output_cols + hidden_cols + down_cols) * 2 * 2
    );
    assert_eq!(
        summary.output_bytes,
        u64::from(qkv_rows + attention_output_rows + gate_up_rows + down_rows) * 2 * 4
    );
    assert_eq!(summary.pack_kernel_launches, 8);
    assert_eq!(summary.projection_kernel_launches, 4);
    assert_eq!(summary.scatter_kernel_launches, 8);
    assert_eq!(summary.dependency_kernel_launches, 12);
    assert!(summary.elapsed_ns > 0);
    assert!(summary.qkv_elapsed_ns > 0);
    assert!(summary.attention_output_elapsed_ns > 0);
    assert!(summary.gate_up_elapsed_ns > 0);
    assert!(summary.down_elapsed_ns > 0);
    assert_eq!(summary.hot_path_allocations, 0);
}

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

fn projection_batch_weight_plan(
    blocks: &[CudaHfDecodeSequenceWeightBlock],
) -> CudaHfDecodeSequenceWeightPlan {
    let weight_bytes = blocks.iter().map(|block| block.bytes).sum();
    CudaHfDecodeSequenceWeightPlan {
        blocks: blocks.len() as u32,
        gpu_resident_blocks: blocks.len() as u32,
        gpu_staged_blocks: 0,
        weight_bytes,
        gpu_resident_weight_bytes: weight_bytes,
        gpu_staged_weight_bytes: 0,
        descriptor_hash: hash_weight_blocks(blocks),
    }
}

fn projection_batch_weight_blocks(
    embeddings: &[u16],
    rms: &[u16],
    attn_matrix: &[u16],
    mlp_matrix: &[u16],
    down_matrix: &[u16],
    lm_head: &[u16],
) -> Vec<CudaHfDecodeSequenceWeightBlock> {
    let sources = [
        embeddings,
        rms,
        attn_matrix,
        attn_matrix,
        attn_matrix,
        attn_matrix,
        rms,
        mlp_matrix,
        mlp_matrix,
        down_matrix,
        rms,
        lm_head,
    ];
    let mut offset_bytes = 0;
    sources
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let bytes = (source.len() * std::mem::size_of::<u16>()) as u64;
            let block = CudaHfDecodeSequenceWeightBlock {
                host_source: source.as_ptr(),
                block_id: index as u64 + 1,
                block_version: 1,
                offset_bytes,
                bytes,
                strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
                reserved: 0,
                ..CudaHfDecodeSequenceWeightBlock::default()
            };
            offset_bytes += bytes;
            block
        })
        .collect()
}
