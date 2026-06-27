use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::footprint::CudaHfDecodeSequenceFootprint;
use crate::decode::hf_sequence::request::{
    CUDA_HF_DECODE_SEQUENCE_DTYPE_F16, CudaHfDecodeSequenceRequest,
};
use crate::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use crate::decode::hf_sequence::weight_plan::{
    CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT, CUDA_HF_WEIGHT_STRATEGY_GPU_STAGED,
    CudaHfDecodeSequenceWeightBlock, CudaHfDecodeSequenceWeightPlan, hash_weight_blocks,
};
use crate::smoke::status::SmokeStatus;
#[test]
fn hf_decode_sequence_summary_serializes_device_token_fields() {
    let summary = CudaHfDecodeSequenceSummary {
        status: SmokeStatus::Ok,
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        vocab_size: 4,
        layer_count: 2,
        steps: 4,
        seed_token: 0,
        tokens: vec![1, 2, 3, 0],
        observed_token_hash: 7,
        planned_footprint: CudaHfDecodeSequenceFootprint::default(),
        device_total_memory_bytes: Some(1024),
        device_free_memory_bytes: Some(512),
        fits_device_free_memory: Some(true),
        resident_weight_bytes: 128,
        planned_weight_blocks: 12,
        planned_gpu_resident_blocks: 6,
        planned_gpu_staged_blocks: 6,
        planned_weight_bytes: 128,
        planned_gpu_resident_weight_bytes: 64,
        planned_gpu_staged_weight_bytes: 64,
        descriptor_gpu_resident_h2d_bytes: 32,
        descriptor_gpu_staged_h2d_bytes: 96,
        planned_weight_descriptor_count: 12,
        planned_weight_descriptor_hash: 123,
        resident_kv_bytes: 64,
        kv_tokens: 4,
        device_arena_bytes: 240,
        pinned_host_bytes: 180,
        h2d_bytes: 160,
        d2h_bytes: 160,
        graph_replays: 4,
        graph_nodes: 1,
        graph_launches: 4,
        graph_captures: 1,
        graph_cache_hits: 0,
        kernel_launches: 4,
        device_elapsed_ns: 900,
        projection_ns: 500,
        qkv_projection_ns: 100,
        attention_output_projection_ns: 80,
        gate_up_projection_ns: 120,
        down_projection_ns: 90,
        lm_head_projection_ns: 110,
        attention_ns: 100,
        mlp_ns: 90,
        norm_ns: 80,
        sampling_ns: 20,
        sync_calls: 1,
        host_causality_edges: 0,
        hot_path_allocations: 0,
        error: None,
    };
    let json = summary.to_json();
    for expected in [
        "\"status\":\"ok\"",
        "\"steps\":4",
        "\"tokens\":[1,2,3,0]",
        "\"projection_ns\":500",
        "\"lm_head_projection_ns\":110",
    ] {
        assert!(json.contains(expected));
    }
}
#[test]
fn hf_decode_sequence_runs_device_first_steps_when_device_is_available() {
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
    let weight_blocks = sequence_weight_blocks(&embeddings, &rms, &matrix, &lm_head);
    let summary = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 2,
        heads: 1,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 2,
        vocab_size: 4,
        steps: 4,
        seed_token: 0,
        prompt_tokens: &[0],
        eos_token: Some(2),
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &embeddings,
        layers: &layers,
        final_norm_weight: &rms,
        lm_head: &lm_head,
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 12,
            gpu_resident_blocks: 6,
            gpu_staged_blocks: 6,
            weight_bytes: 100,
            gpu_resident_weight_bytes: 52,
            gpu_staged_weight_bytes: 48,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
    }
    .run();
    if summary.status != SmokeStatus::Ok {
        return;
    }
    assert_eq!(summary.tokens, vec![1, 2]);
    assert_eq!(summary.graph_replays, 4);
    assert!(summary.resident_kv_bytes > 0);
    assert_eq!(summary.kv_tokens, 2);
    assert_eq!(summary.graph_launches, 4);
    assert_eq!((summary.graph_nodes, summary.kernel_launches), (3, 12));
    assert_eq!(summary.sync_calls, 1);
    assert!(summary.planned_footprint.context_tokens >= summary.kv_tokens);
    assert_eq!(summary.fits_device_free_memory, Some(true));
    assert_eq!(summary.host_causality_edges, 0);
    assert_eq!(summary.hot_path_allocations, 0);
    assert_eq!(summary.planned_weight_blocks, 12);
    assert_eq!(summary.planned_weight_bytes, summary.resident_weight_bytes);
    assert_eq!(summary.planned_gpu_resident_weight_bytes, 52);
    assert_eq!(summary.planned_gpu_staged_weight_bytes, 48);
    assert_eq!(summary.descriptor_gpu_resident_h2d_bytes, 52);
    assert_eq!(summary.descriptor_gpu_staged_h2d_bytes, 48);
    assert_eq!(summary.planned_weight_descriptor_count, 12);
    assert_eq!(
        summary.planned_weight_descriptor_hash,
        hash_weight_blocks(&weight_blocks)
    );
}
fn sequence_weight_blocks(
    embeddings: &[u16],
    rms: &[u16],
    matrix: &[u16],
    lm_head: &[u16],
) -> Vec<CudaHfDecodeSequenceWeightBlock> {
    let bytes = [16, 4, 8, 8, 8, 8, 4, 8, 8, 8, 4, 16];
    let sources = [
        embeddings, rms, matrix, matrix, matrix, matrix, rms, matrix, matrix, matrix, rms, lm_head,
    ];
    let mut offset_bytes = 0;
    bytes
        .iter()
        .zip(sources)
        .enumerate()
        .map(|(index, (bytes, source))| {
            let strategy = if index < 6 {
                CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT
            } else {
                CUDA_HF_WEIGHT_STRATEGY_GPU_STAGED
            };
            let block = CudaHfDecodeSequenceWeightBlock {
                host_source: source.as_ptr(),
                block_id: index as u64 + 1,
                block_version: 0,
                offset_bytes,
                bytes: *bytes,
                strategy,
                reserved: 0,
            };
            offset_bytes += *bytes;
            block
        })
        .collect()
}
