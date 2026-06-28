use std::ffi::CString;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::request::{
    CUDA_HF_DECODE_SEQUENCE_DTYPE_F16, CudaHfDecodeSequenceRequest,
};
use crate::decode::hf_sequence::weight_plan::{
    CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT, CUDA_HF_WEIGHT_STRATEGY_GPU_STAGED,
    CudaHfDecodeSequenceWeightBlock, CudaHfDecodeSequenceWeightPlan, hash_weight_blocks,
};
use crate::smoke::status::SmokeStatus;

#[test]
fn declared_weight_descriptors_stream_file_backed_sources() {
    let _guard = super::cuda_test_lock();

    if crate::smoke::probe::smoke().status != SmokeStatus::Ok {
        return;
    }
    let one = 0x3c00;
    let zero = 0x0000;
    let neg_one = 0xbc00;
    let embeddings = [one, zero, zero, one, neg_one, zero, zero, neg_one];
    let rms = [one, one];
    let matrix = [zero; 4];
    let lm_head = [zero, neg_one, one, zero, zero, one, neg_one, zero];
    let (path, source_path, weight_blocks) =
        file_backed_sequence_weight_blocks(&embeddings, &rms, &matrix, &lm_head);
    let _keep_path_alive = source_path;
    let layer = descriptor_marker_layer();
    let layers = [layer];
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
        eos_token: None,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
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
    let _ = fs::remove_file(path);

    if summary.status != SmokeStatus::Ok {
        assert_eq!(summary.status, SmokeStatus::Unavailable);
        return;
    }
    assert_eq!(summary.tokens, vec![1, 2, 3, 0]);
    assert_eq!(summary.descriptor_gpu_resident_h2d_bytes, 52);
    assert_eq!(summary.descriptor_gpu_staged_h2d_bytes, 48);
    assert!(summary.sync_calls > 1);
}

fn descriptor_marker_layer() -> CudaHfDecodeChainLayer<'static> {
    CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_k: &[],
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &[],
        w_o: &[],
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &[],
        w_up: &[],
        w_down: &[],
    }
}

fn file_backed_sequence_weight_blocks(
    embeddings: &[u16],
    rms: &[u16],
    matrix: &[u16],
    lm_head: &[u16],
) -> (
    std::path::PathBuf,
    CString,
    Vec<CudaHfDecodeSequenceWeightBlock>,
) {
    let path = std::env::temp_dir().join(format!(
        "nerva-cuda-file-backed-descriptor-{}-{}.bin",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let source_path = CString::new(path.as_os_str().as_bytes()).unwrap();
    let bytes = [16, 4, 8, 8, 8, 8, 4, 8, 8, 8, 4, 16];
    let sources = [
        embeddings, rms, matrix, matrix, matrix, matrix, rms, matrix, matrix, matrix, rms, lm_head,
    ];
    let mut file_bytes = Vec::from([0x51u8; 16]);
    let mut file_offsets = Vec::with_capacity(sources.len());
    for source in sources {
        file_offsets.push(file_bytes.len() as u64);
        append_u16_bytes(&mut file_bytes, source);
    }
    fs::write(&path, file_bytes).unwrap();
    let blocks = descriptor_blocks(&bytes, &file_offsets, &source_path);
    (path, source_path, blocks)
}

fn descriptor_blocks(
    bytes: &[u64; 12],
    file_offsets: &[u64],
    source_path: &CString,
) -> Vec<CudaHfDecodeSequenceWeightBlock> {
    let mut offset_bytes = 0;
    bytes
        .iter()
        .enumerate()
        .map(|(index, bytes)| {
            let strategy = if index < 6 {
                CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT
            } else {
                CUDA_HF_WEIGHT_STRATEGY_GPU_STAGED
            };
            let block = CudaHfDecodeSequenceWeightBlock {
                source_file: source_path.as_ptr(),
                source_file_len: source_path.as_bytes().len() as u64,
                file_offset_begin: file_offsets[index],
                block_id: index as u64 + 1,
                block_version: 0,
                offset_bytes,
                bytes: *bytes,
                strategy,
                reserved: 0,
                ..CudaHfDecodeSequenceWeightBlock::default()
            };
            offset_bytes += *bytes;
            block
        })
        .collect()
}

fn append_u16_bytes(out: &mut Vec<u8>, values: &[u16]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}
