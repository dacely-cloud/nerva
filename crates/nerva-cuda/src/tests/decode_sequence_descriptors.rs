use crate::decode::hf_chain::layer::{
    CUDA_HF_ATTENTION_DEEPSEEK_MLA, CUDA_HF_ATTENTION_LINEAR_GDN, CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR,
    CUDA_HF_DEEPSEEK_FLAG_HASH_ROUTER, CUDA_HF_DEEPSEEK_FLAG_MOE,
    CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS, CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER,
    CUDA_HF_DEEPSEEK_MODE_V3_MLA, CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED,
    CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER, CUDA_HF_DEEPSEEK_MODE_V4_SWA,
    CUDA_HF_DEEPSEEK_MODE_V32_MLA_INDEXER, CUDA_HF_DEEPSEEK_ROPE_SCALING_DEEPSEEK,
    CUDA_HF_DEEPSEEK_ROPE_SCALING_NONE, CUDA_HF_MLP_DENSE, CUDA_HF_MLP_SPARSE_MOE,
    CudaHfDecodeChainLayer, CudaHfDeepSeekLayer, CudaHfLinearGdnLayer,
};
use crate::decode::hf_sequence::footprint::estimate_sequence_footprint;
use crate::decode::hf_sequence::layout_plan::{
    CUDA_HF_SEQUENCE_MISSING_OFFSET, CudaHfDecodeSequenceLayoutPlan,
    CudaHfDecodeSequenceLayoutPlanRequest,
};
use crate::decode::hf_sequence::request::{
    CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16, CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
    CudaHfDecodeSamplerConfig, CudaHfDecodeSequenceRequest,
};
use crate::decode::hf_sequence::session::request::{
    CUDA_HF_DEEPSEEK_V4_MHC_STATE_COMB_MIX, CUDA_HF_DEEPSEEK_V4_MHC_STATE_POST_MIX,
    CUDA_HF_DEEPSEEK_V4_MHC_STATE_RESIDUAL, CudaHfDecodeSequenceExperimentalRtConfig,
    CudaHfDecodeSequenceSessionConfig, CudaHfDecodeSequenceSessionCreateOutput,
};
use crate::decode::hf_sequence::session::stateful::CudaHfDecodeSequenceLoop;
use crate::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use crate::decode::hf_sequence::weight_plan::{
    CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT, CudaHfDecodeSequenceWeightBlock,
    CudaHfDecodeSequenceWeightPlan, hash_weight_blocks,
};
use crate::smoke::status::SmokeStatus;

use super::decode_sequence_descriptor_blocks::{
    run_null_legacy_descriptor_decode, tiny_descriptor_weights,
};

fn fnv_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x00000100000001b3);
    }
    hash
}

fn encode_e8m0_scale(scale: f32) -> u8 {
    (scale.log2().ceil() as i32 + 127).clamp(0, 255) as u8
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f32_values_from_le_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn f32_to_f8_e4m3fn_bits_nearest(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let mut best_bits = 0u8;
    let mut best_error = f32::INFINITY;
    for bits in 0u8..=254 {
        let candidate = f8_e4m3fn_bits_to_f32(bits);
        if candidate.is_nan() {
            continue;
        }
        let error = (candidate - value).abs();
        if error < best_error || (error == best_error && bits < best_bits) {
            best_error = error;
            best_bits = bits;
        }
    }
    best_bits
}

fn f8_e4m3fn_bits_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exp = (bits >> 3) & 0x0f;
    let frac = bits & 0x07;
    if exp == 0 {
        if frac == 0 {
            return sign * 0.0;
        }
        return sign * ((frac as f32) * 0.125) * 2.0f32.powi(-6);
    }
    if exp == 0x0f && frac == 0x07 {
        return f32::NAN;
    }
    sign * (1.0 + (frac as f32) * 0.125) * 2.0f32.powi(exp as i32 - 7)
}

fn deepseek_sparse_attention_output_hash_head0(values: &[f32]) -> u64 {
    values
        .iter()
        .enumerate()
        .fold(0u64, |acc, (position, value)| {
            let active_head = ((position as u64) + 1) * 1_315_423_911u64
                ^ 2_654_435_761u64
                ^ 97_531u64
                ^ f32_to_bf16_bits(*value) as u64;
            let zero_head =
                ((position as u64) + 1) * 1_315_423_911u64 ^ (2u64 * 2_654_435_761u64) ^ 97_531u64;
            acc + active_head + zero_head
        })
}

fn deepseek_sparse_topk_selection_hash(selections: &[(usize, &[u64])]) -> u64 {
    selections.iter().fold(0u64, |acc, (position, slots)| {
        acc + slots.iter().enumerate().fold(0u64, |inner, (rank, slot)| {
            inner
                + (((*position as u64) + 1) * 1_315_423_911u64
                    ^ ((rank as u64) + 1) * 2_654_435_761u64
                    ^ (*slot + 1))
        })
    })
}

fn deepseek_rope_value_reference(
    left: f32,
    right: f32,
    offset: usize,
    dim: usize,
    position: usize,
    theta: f32,
    second: bool,
) -> f32 {
    if theta <= 0.0 || dim < 2 {
        return if second { right } else { left };
    }
    let exponent = (2 * offset) as f32 / dim as f32;
    let angle = position as f32 / theta.powf(exponent);
    let sin_value = angle.sin();
    let cos_value = angle.cos();
    if second {
        right * cos_value + left * sin_value
    } else {
        left * cos_value - right * sin_value
    }
}

fn fullsize_v4_swa_expected_token(position: usize, normalized_k: f32) -> Vec<f32> {
    let qk_nope = 448usize;
    let qk_rope = 64usize;
    let mut values = vec![normalized_k; qk_nope + qk_rope];
    let rope_half = qk_rope / 2;
    for offset in 0..rope_half {
        let left = qk_nope + offset;
        let right = left + rope_half;
        let left_value = values[left];
        let right_value = values[right];
        values[left] = deepseek_rope_value_reference(
            left_value,
            right_value,
            offset,
            qk_rope,
            position,
            10_000.0,
            false,
        );
        values[right] = deepseek_rope_value_reference(
            left_value,
            right_value,
            offset,
            qk_rope,
            position,
            10_000.0,
            true,
        );
    }
    values
}

fn assert_page_bytes_eq(actual: &[u8], expected: &[u8], message: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{message}: page lengths differ"
    );
    if let Some(index) = actual
        .iter()
        .zip(expected.iter())
        .position(|(actual, expected)| actual != expected)
    {
        let start = index.saturating_sub(8);
        let end = (index + 9).min(actual.len());
        panic!(
            "{message}: byte {index} differs actual={} expected={} actual_window={:?} expected_window={:?}",
            actual[index],
            expected[index],
            &actual[start..end],
            &expected[start..end]
        );
    }
}

fn expected_deepseek_v4_fp8_ds_mla_page(
    token_kv: &[Vec<f32>],
    qk_nope: usize,
    qk_rope: usize,
    page_bytes: usize,
    block_tokens: usize,
) -> Vec<u8> {
    let token_stride = qk_nope + qk_rope * 2;
    let scale_dim = qk_nope / 64 + 1;
    let scale_base = block_tokens * token_stride;
    let mut expected = vec![0u8; page_bytes];
    for (token, kv) in token_kv.iter().enumerate() {
        assert!(token < block_tokens);
        assert_eq!(kv.len(), qk_nope + qk_rope);
        let data_base = token * token_stride;
        for scale_index in 0..scale_dim {
            let start = scale_index * 64;
            let end = (start + 64).min(qk_nope);
            let mut absmax = 0.0f32;
            for value in kv.iter().take(end).skip(start) {
                let quant_input = bf16_to_f32(f32_to_bf16_bits(*value));
                absmax = absmax.max(quant_input.abs());
            }
            let scale = (absmax.max(1.0e-4) / 448.0).log2().ceil().exp2();
            expected[scale_base + token * scale_dim + scale_index] = if start < qk_nope {
                encode_e8m0_scale(scale)
            } else {
                0
            };
            for (dim, value) in kv.iter().enumerate().take(end).skip(start) {
                let quant_input = bf16_to_f32(f32_to_bf16_bits(*value));
                let scaled = (quant_input / scale).clamp(-448.0, 448.0);
                expected[data_base + dim] = f32_to_f8_e4m3fn_bits_nearest(scaled);
            }
        }
        for (dim, value) in kv.iter().enumerate().take(qk_nope + qk_rope).skip(qk_nope) {
            let bits = f32_to_bf16_bits(*value);
            let rope_local = dim - qk_nope;
            let offset = data_base + qk_nope + rope_local * 2;
            expected[offset] = (bits & 0xff) as u8;
            expected[offset + 1] = (bits >> 8) as u8;
        }
    }
    expected
}

fn expected_deepseek_v4_swa_fp8_ds_mla_page(
    token_kv: &[Vec<f32>],
    qk_nope: usize,
    qk_rope: usize,
    page_bytes: usize,
) -> Vec<u8> {
    expected_deepseek_v4_fp8_ds_mla_page(token_kv, qk_nope, qk_rope, page_bytes, 64)
}

fn expected_zero_deepseek_v4_swa_fp8_ds_mla_page(
    written_tokens: usize,
    qk_nope: usize,
    qk_rope: usize,
    page_bytes: usize,
) -> Vec<u8> {
    expected_deepseek_v4_swa_fp8_ds_mla_page(
        &vec![vec![0.0; qk_nope + qk_rope]; written_tokens],
        qk_nope,
        qk_rope,
        page_bytes,
    )
}

fn descriptor_source_element(arena_offset: u64, hidden: usize, vocab_size: usize) -> usize {
    let offset = arena_offset as usize;
    let embedding_elements = hidden * vocab_size;
    if offset >= embedding_elements + hidden * 2 {
        offset - hidden * 2
    } else {
        offset
    }
}

fn write_descriptor_u16(
    storage: &mut [u16],
    arena_offset: u64,
    hidden: usize,
    vocab_size: usize,
    value: u16,
) {
    let index = descriptor_source_element(arena_offset, hidden, vocab_size);
    storage[index] = value;
}

fn write_descriptor_byte(
    storage: &mut [u16],
    arena_offset: u64,
    byte_offset: usize,
    hidden: usize,
    vocab_size: usize,
    value: u8,
) {
    let base = descriptor_source_element(arena_offset, hidden, vocab_size) * 2;
    let absolute = base + byte_offset;
    let slot = absolute / 2;
    if absolute % 2 == 0 {
        storage[slot] = (storage[slot] & 0xff00) | value as u16;
    } else {
        storage[slot] = (storage[slot] & 0x00ff) | ((value as u16) << 8);
    }
}

fn write_arena_byte(storage: &mut [u16], arena_offset: u64, byte_offset: usize, value: u8) {
    let absolute = arena_offset as usize * 2 + byte_offset;
    let slot = absolute / 2;
    if absolute % 2 == 0 {
        storage[slot] = (storage[slot] & 0xff00) | value as u16;
    } else {
        storage[slot] = (storage[slot] & 0x00ff) | ((value as u16) << 8);
    }
}

fn read_arena_byte(storage: &[u16], arena_offset: u64, byte_offset: usize) -> u8 {
    let absolute = arena_offset as usize * 2 + byte_offset;
    let slot = absolute / 2;
    if absolute % 2 == 0 {
        (storage[slot] & 0x00ff) as u8
    } else {
        (storage[slot] >> 8) as u8
    }
}

fn write_arena_nibble(
    storage: &mut [u16],
    arena_offset: u64,
    byte_offset: usize,
    high: bool,
    nibble: u8,
) {
    let current = read_arena_byte(storage, arena_offset, byte_offset);
    let value = if high {
        (current & 0x0f) | ((nibble & 0x0f) << 4)
    } else {
        (current & 0xf0) | (nibble & 0x0f)
    };
    write_arena_byte(storage, arena_offset, byte_offset, value);
}

fn write_arena_f32(storage: &mut [u16], arena_offset: u64, value: f32) {
    let bits = value.to_bits();
    storage[arena_offset as usize] = bits as u16;
    storage[arena_offset as usize + 1] = (bits >> 16) as u16;
}

fn write_arena_u64(storage: &mut [u16], arena_offset: u64, index: usize, value: u64) {
    let offset = arena_offset as usize + index * 4;
    storage[offset] = value as u16;
    storage[offset + 1] = (value >> 16) as u16;
    storage[offset + 2] = (value >> 32) as u16;
    storage[offset + 3] = (value >> 48) as u16;
}

fn write_arena_mxfp4_rank3_value(
    storage: &mut [u16],
    arena_offset: u64,
    rows: usize,
    packed_cols: usize,
    expert: usize,
    row: usize,
    col: usize,
    nibble: u8,
) {
    let packed_col = col / 2;
    let byte_offset = (expert * rows + row) * packed_cols + packed_col;
    write_arena_nibble(storage, arena_offset, byte_offset, col % 2 == 1, nibble);
}

fn rank3_byte_slots(depth: usize, rows: usize, cols: usize) -> u64 {
    (depth * rows * cols).div_ceil(2) as u64
}

fn byte_slots(rows: usize, cols: usize) -> u64 {
    (rows * cols).div_ceil(2) as u64
}

fn write_arena_mxfp4_rank3_scales(
    storage: &mut [u16],
    arena_offset: u64,
    rows: usize,
    packed_cols: usize,
    expert: usize,
    scale_byte: u8,
) {
    let scale_cols = packed_cols.div_ceil(16);
    for row in 0..rows {
        for scale_col in 0..scale_cols {
            let byte_offset = (expert * rows + row) * scale_cols + scale_col;
            write_arena_byte(storage, arena_offset, byte_offset, scale_byte);
        }
    }
}

fn descriptor_weight_blocks<'a>(
    storage: &'a [u16],
    hidden: usize,
    vocab_size: usize,
    resident_weight_bytes: u64,
) -> Vec<CudaHfDecodeSequenceWeightBlock> {
    let embedding_elements = hidden * vocab_size;
    let embedding_bytes = (embedding_elements * core::mem::size_of::<u16>()) as u64;
    assert!(resident_weight_bytes >= embedding_bytes);
    let mut blocks = vec![CudaHfDecodeSequenceWeightBlock {
        host_source: storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: embedding_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    if resident_weight_bytes > embedding_bytes {
        blocks.push(CudaHfDecodeSequenceWeightBlock {
            host_source: unsafe { storage.as_ptr().add(embedding_elements) },
            source_file: core::ptr::null(),
            source_file_len: 0,
            file_offset_begin: 0,
            block_id: 2,
            block_version: 1,
            offset_bytes: embedding_bytes,
            bytes: resident_weight_bytes - embedding_bytes,
            strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
            reserved: 0,
        });
    }
    blocks
}

#[test]
fn linear_gdn_layer_validation_preserves_layout_metadata() {
    let hidden = 4;
    let rms = vec![0x3c00; hidden];
    let router = vec![0x3c00; 4 * hidden];
    let expert_gate_up = vec![0x3c00; 4 * 2 * 3 * hidden];
    let expert_down = vec![0x3c00; 4 * hidden * 3];
    let linear_conv = vec![0x3c00; 28];
    let linear_qkv = vec![0x3c00; 28];
    let linear_z = vec![0x3c00; 12];
    let linear_b = vec![0x3c00; 4];
    let linear_a = vec![0x3c00; 4];
    let linear_dt_bias = vec![0x3c00; 1];
    let linear_a_log = vec![0.0f32];
    let linear_norm = vec![0x0000, 0x3f80, 0x0000, 0x3f80, 0x0000, 0x3f80];
    let linear_out = vec![0x3c00; 12];
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &[],
        w_q_gate: None,
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
        w_router: Some(&router),
        w_expert_gate_up: Some(&expert_gate_up),
        w_expert_down: Some(&expert_down),
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: Some(CudaHfLinearGdnLayer {
            key_heads: 1,
            value_heads: 1,
            key_head_dim: 2,
            value_head_dim: 3,
            conv_kernel: 4,
            w_conv: &linear_conv,
            w_qkv: &linear_qkv,
            w_z: &linear_z,
            w_b: &linear_b,
            w_a: &linear_a,
            dt_bias: &linear_dt_bias,
            a_log: &linear_a_log,
            norm_weight: &linear_norm,
            w_out: &linear_out,
        }),
        deepseek: None,
        mlp_kind: CUDA_HF_MLP_SPARSE_MOE,
        moe_intermediate: 3,
        shared_expert_intermediate: 0,
        num_experts: 4,
        experts_per_token: 2,
        norm_topk_prob: true,
        attention_kind: CUDA_HF_ATTENTION_LINEAR_GDN,
    };

    assert_eq!(layer.validate(hidden, 4, 4, 2, 8), None);
    let ffi = layer.to_ffi();
    assert_eq!(ffi.linear_key_heads, 1);
    assert_eq!(ffi.linear_value_head_dim, 3);
    assert!(!ffi.w_linear_a_log.is_null());
}

#[test]
fn deepseek_mla_layer_validation_preserves_layout_metadata() {
    let hidden = 4096;
    let rms = vec![0x3c00; hidden];
    let deepseek = CudaHfDeepSeekLayer {
        mode: CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER,
        flags: CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER
            | CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR
            | CUDA_HF_DEEPSEEK_FLAG_HASH_ROUTER
            | CUDA_HF_DEEPSEEK_FLAG_MOE,
        hc_mult: 4,
        hc_sinkhorn_iters: 20,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        o_lora_rank: 1536,
        o_groups: 8,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
        compress_ratio: 4,
        index_topk: 2048,
        index_n_heads: 64,
        index_head_dim: 128,
        router_num_groups: 0,
        router_topk_groups: 0,
        routed_scaling_factor: 1.0,
        hc_eps: 1.0e-6,
        hc_post_alpha: 2.0,
        rope_scaling_type: CUDA_HF_DEEPSEEK_ROPE_SCALING_DEEPSEEK,
        rope_original_max_position: 4096,
        rope_scaling_factor: 40.0,
        rope_extrapolation_factor: 1.0,
        rope_attn_factor: 1.0,
        rope_beta_fast: 32.0,
        rope_beta_slow: 1.0,
        rope_mscale: 1.0,
        rope_mscale_all_dim: 0.0,
        compress_rope_theta: Some(1_000_000.0),
        swiglu_limit: Some(10.0),
    };
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &[],
        w_q_gate: None,
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
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: Some(deepseek),
        mlp_kind: CUDA_HF_MLP_SPARSE_MOE,
        moe_intermediate: 2048,
        shared_expert_intermediate: 0,
        num_experts: 128,
        experts_per_token: 8,
        norm_topk_prob: true,
        attention_kind: CUDA_HF_ATTENTION_DEEPSEEK_MLA,
    };

    assert_eq!(layer.validate(hidden, hidden, 512, 128, 4096), None);

    let ffi = layer.to_ffi();
    assert_eq!(ffi.attention_kind, CUDA_HF_ATTENTION_DEEPSEEK_MLA);
    assert_eq!(ffi.deepseek_mode, deepseek.mode);
    assert_eq!(ffi.deepseek_flags, deepseek.flags);
    assert_eq!(ffi.deepseek_hc_mult, deepseek.hc_mult as u32);
    assert_eq!(
        ffi.deepseek_hc_sinkhorn_iters,
        deepseek.hc_sinkhorn_iters as u32
    );
    assert_eq!(ffi.deepseek_q_lora_rank, deepseek.q_lora_rank as u32);
    assert_eq!(ffi.deepseek_kv_lora_rank, deepseek.kv_lora_rank as u32);
    assert_eq!(ffi.deepseek_o_lora_rank, deepseek.o_lora_rank as u32);
    assert_eq!(ffi.deepseek_o_groups, deepseek.o_groups as u32);
    assert_eq!(
        ffi.deepseek_qk_nope_head_dim,
        deepseek.qk_nope_head_dim as u32
    );
    assert_eq!(
        ffi.deepseek_qk_rope_head_dim,
        deepseek.qk_rope_head_dim as u32
    );
    assert_eq!(ffi.deepseek_v_head_dim, deepseek.v_head_dim as u32);
    assert_eq!(ffi.deepseek_compress_ratio, deepseek.compress_ratio as u32);
    assert_eq!(ffi.deepseek_index_topk, deepseek.index_topk as u32);
    assert_eq!(ffi.deepseek_index_n_heads, deepseek.index_n_heads as u32);
    assert_eq!(ffi.deepseek_index_head_dim, deepseek.index_head_dim as u32);
    assert_eq!(
        ffi.deepseek_router_num_groups,
        deepseek.router_num_groups as u32
    );
    assert_eq!(
        ffi.deepseek_router_topk_groups,
        deepseek.router_topk_groups as u32
    );
    assert_eq!(
        ffi.deepseek_routed_scaling_factor,
        deepseek.routed_scaling_factor
    );
    assert_eq!(ffi.deepseek_hc_eps, deepseek.hc_eps);
    assert_eq!(ffi.deepseek_hc_post_alpha, deepseek.hc_post_alpha);
    assert_eq!(ffi.deepseek_rope_scaling_type, deepseek.rope_scaling_type);
    assert_eq!(
        ffi.deepseek_rope_original_max_position,
        deepseek.rope_original_max_position as u32
    );
    assert_eq!(
        ffi.deepseek_rope_scaling_factor,
        deepseek.rope_scaling_factor
    );
    assert_eq!(ffi.deepseek_rope_mscale, deepseek.rope_mscale);
    assert_eq!(
        ffi.deepseek_compress_rope_theta,
        deepseek.compress_rope_theta.unwrap_or(0.0)
    );
    assert_eq!(
        ffi.deepseek_swiglu_limit,
        deepseek.swiglu_limit.unwrap_or(0.0)
    );

    let descriptor = layer.to_descriptor_layout_ffi();
    assert!(descriptor.w_q.is_null());
    assert!(descriptor.w_gate.is_null());
    assert_eq!(descriptor.deepseek_mode, deepseek.mode);
    assert_eq!(descriptor.deepseek_flags, deepseek.flags);
    assert_eq!(
        descriptor.deepseek_hc_sinkhorn_iters,
        deepseek.hc_sinkhorn_iters as u32
    );
    assert_eq!(
        descriptor.deepseek_router_num_groups,
        deepseek.router_num_groups as u32
    );
    assert_eq!(descriptor.deepseek_hc_eps, deepseek.hc_eps);
    assert_eq!(descriptor.deepseek_hc_post_alpha, deepseek.hc_post_alpha);
    assert_eq!(
        descriptor.deepseek_rope_scaling_type,
        deepseek.rope_scaling_type
    );
    assert_eq!(
        descriptor.deepseek_rope_original_max_position,
        deepseek.rope_original_max_position as u32
    );
    assert_eq!(
        descriptor.deepseek_rope_scaling_factor,
        deepseek.rope_scaling_factor
    );
    assert_eq!(descriptor.deepseek_rope_mscale, deepseek.rope_mscale);
    assert_eq!(
        descriptor.deepseek_compress_rope_theta,
        deepseek.compress_rope_theta.unwrap_or(0.0)
    );
    assert_eq!(
        descriptor.deepseek_swiglu_limit,
        deepseek.swiglu_limit.unwrap_or(0.0)
    );
}

#[test]
fn deepseek_v3_mla_shape_matches_vllm_contract() {
    let deepseek = CudaHfDeepSeekLayer {
        mode: CUDA_HF_DEEPSEEK_MODE_V3_MLA,
        flags: 0,
        hc_mult: 0,
        hc_sinkhorn_iters: 0,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        o_lora_rank: 0,
        o_groups: 0,
        qk_nope_head_dim: 128,
        qk_rope_head_dim: 64,
        v_head_dim: 128,
        compress_ratio: 1,
        index_topk: 0,
        index_n_heads: 0,
        index_head_dim: 0,
        router_num_groups: 8,
        router_topk_groups: 4,
        routed_scaling_factor: 2.5,
        hc_eps: 0.0,
        hc_post_alpha: 0.0,
        rope_scaling_type: CUDA_HF_DEEPSEEK_ROPE_SCALING_NONE,
        rope_original_max_position: 0,
        rope_scaling_factor: 0.0,
        rope_extrapolation_factor: 1.0,
        rope_attn_factor: 1.0,
        rope_beta_fast: 32.0,
        rope_beta_slow: 1.0,
        rope_mscale: 1.0,
        rope_mscale_all_dim: 0.0,
        compress_rope_theta: None,
        swiglu_limit: None,
    };

    assert!(deepseek.is_v3_mla());
    assert!(!deepseek.is_v4_mla());
    assert_eq!(deepseek.qk_head_dim(), Some(192));

    let shape = deepseek
        .v3_mla_shape(128)
        .expect("DeepSeek V3 dimensions should form an MLA shape");
    assert_eq!(shape.num_heads, 128);
    assert_eq!(shape.qk_head_dim, 192);
    assert_eq!(shape.q_rows, 24_576);
    assert_eq!(shape.kv_cache_width, 576);
    assert_eq!(shape.kv_b_rows, 32_768);
    assert_eq!(shape.value_rows, 16_384);
}

#[test]
fn deepseek_v3_mla_snapshot_matches_vllm_latent_cache_row() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let hidden = 4usize;
    let heads = 2usize;
    let kv_heads = 1usize;
    let head_dim = 2usize;
    let intermediate = 4usize;
    let vocab_size = 8usize;
    let layer = tiny_deepseek_v3_descriptor_layer();
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: hidden as u32,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        intermediate: intermediate as u32,
        vocab_size: vocab_size as u32,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V3 MLA descriptor layer");
    assert_eq!(plan.deepseek_kv_cache_width, 3);
    assert_ne!(plan.w_k, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.deepseek_kv_a_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.k_norm, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    for dim in 0..hidden {
        weight_storage[dim] = f32_to_bf16_bits(1.0);
        weight_storage[plan.rms_attn as usize + dim] = f32_to_bf16_bits(1.0);
    }
    for dim in 0..2usize {
        weight_storage[plan.k_norm as usize + dim] = f32_to_bf16_bits(1.0);
    }
    let one_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
    for row in 0..3usize {
        let absolute = plan.w_k as usize * 2 + row * hidden;
        let slot = absolute / 2;
        if absolute % 2 == 0 {
            weight_storage[slot] = (weight_storage[slot] & 0xff00) | one_fp8 as u16;
        } else {
            weight_storage[slot] = (weight_storage[slot] & 0x00ff) | ((one_fp8 as u16) << 8);
        }
    }
    let scale_bits = 1.0f32.to_bits();
    weight_storage[plan.deepseek_kv_a_scale as usize] = scale_bits as u16;
    weight_storage[plan.deepseek_kv_a_scale as usize + 1] = (scale_bits >> 16) as u16;

    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16,
        hidden,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        vocab_size,
        max_context_tokens: 4,
        rms_eps: 1.0e-5,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: CudaHfDecodeSequenceExperimentalRtConfig::default(),
    };
    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3 MLA session should create before cache snapshot: {:?}",
        created.summary.error
    );
    let mut session = created.session.expect("V3 MLA session handle should exist");
    let summary = session.run(&[0], 1, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V3 MLA one-token run should populate the latent cache: {:?}",
        summary.error
    );

    let snapshot = session.deepseek_v3_mla_kv_snapshot(0, 96);
    assert_eq!(
        snapshot.status,
        SmokeStatus::Ok,
        "V3 MLA KV snapshot should copy device cache: {:?}",
        snapshot.error
    );
    assert_eq!(snapshot.block_count, 1);
    assert_eq!(snapshot.layer_offset_bytes, 0);
    assert_eq!(snapshot.layer_bytes, 96);
    assert_eq!(snapshot.page_bytes, 96);
    assert_eq!(snapshot.copied_bytes, 96);

    let one = f32_to_bf16_bits(1.0).to_le_bytes();
    let mut expected = vec![0u8; 96];
    expected[0..2].copy_from_slice(&one);
    expected[2..4].copy_from_slice(&one);
    expected[4..6].copy_from_slice(&one);
    assert_page_bytes_eq(
        &snapshot.bytes,
        &expected,
        "V3 MLA latent cache row must match vLLM [kv_c, k_pe] ordering",
    );
    assert_eq!(snapshot.output_hash, fnv_hash_bytes(&expected));
}

#[test]
fn deepseek_v3_mla_snapshot_matches_fullsize_vllm_cache_page() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let hidden = 4usize;
    let heads = 1usize;
    let kv_heads = 1usize;
    let qk_nope = 128usize;
    let qk_rope = 64usize;
    let head_dim = qk_nope + qk_rope;
    let kv_lora = 512usize;
    let v_head = 128usize;
    let intermediate = 4usize;
    let vocab_size = 8usize;
    let mut layer = tiny_deepseek_v3_descriptor_layer();
    let deepseek = layer
        .deepseek
        .as_mut()
        .expect("tiny DeepSeek V3 layer should carry DeepSeek metadata");
    deepseek.q_lora_rank = 2;
    deepseek.kv_lora_rank = kv_lora;
    deepseek.qk_nope_head_dim = qk_nope;
    deepseek.qk_rope_head_dim = qk_rope;
    deepseek.v_head_dim = v_head;
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: hidden as u32,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        intermediate: intermediate as u32,
        vocab_size: vocab_size as u32,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept full-size V3 MLA cache dimensions");
    assert_eq!(plan.deepseek_kv_cache_width, (kv_lora + qk_rope) as u32);
    assert_eq!(plan.deepseek_qk_head_dim, head_dim as u32);
    assert_eq!(plan.deepseek_q_rows, head_dim as u32);
    assert_eq!(plan.deepseek_kv_b_rows, (qk_nope + v_head) as u32);
    assert_eq!(plan.deepseek_value_rows, v_head as u32);
    assert_ne!(plan.w_k, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.deepseek_kv_a_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.rms_attn, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.k_norm, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    for dim in 0..hidden {
        weight_storage[dim] = f32_to_bf16_bits(1.0);
        weight_storage[plan.rms_attn as usize + dim] = f32_to_bf16_bits(1.0);
    }
    for dim in 0..kv_lora {
        weight_storage[plan.k_norm as usize + dim] = f32_to_bf16_bits(1.0);
    }
    let one_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
    for row in 0..(kv_lora + qk_rope) {
        write_arena_byte(&mut weight_storage, plan.w_k, row * hidden, one_fp8);
    }
    let scale_blocks = (kv_lora + qk_rope).div_ceil(128);
    for block in 0..scale_blocks {
        write_arena_f32(
            &mut weight_storage,
            plan.deepseek_kv_a_scale + (block * 2) as u64,
            1.0,
        );
    }

    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16,
        hidden,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        vocab_size,
        max_context_tokens: 4,
        rms_eps: 1.0e-5,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: CudaHfDecodeSequenceExperimentalRtConfig::default(),
    };
    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3 MLA full-size cache session should create: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V3 MLA full-size cache session handle should exist");
    let summary = session.run(&[0], 1, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V3 MLA one-token run should populate the full-size latent cache: {:?}",
        summary.error
    );

    let page_bytes = 16 * (kv_lora + qk_rope) * core::mem::size_of::<u16>();
    let snapshot = session.deepseek_v3_mla_kv_snapshot(0, page_bytes);
    assert_eq!(
        snapshot.status,
        SmokeStatus::Ok,
        "V3 MLA full-size KV snapshot should copy device cache: {:?}",
        snapshot.error
    );
    assert_eq!(snapshot.block_count, 1);
    assert_eq!(snapshot.layer_offset_bytes, 0);
    assert_eq!(snapshot.layer_bytes, page_bytes as u64);
    assert_eq!(snapshot.page_bytes, page_bytes as u64);
    assert_eq!(snapshot.copied_bytes, page_bytes as u64);

    let one = f32_to_bf16_bits(1.0).to_le_bytes();
    let mut expected = vec![0u8; page_bytes];
    for dim in 0..(kv_lora + qk_rope) {
        let offset = dim * core::mem::size_of::<u16>();
        expected[offset..offset + 2].copy_from_slice(&one);
    }
    assert_page_bytes_eq(
        &snapshot.bytes,
        &expected,
        "V3 MLA full-size latent cache page must match vLLM [kv_c, k_pe] ordering",
    );
    assert_eq!(snapshot.output_hash, fnv_hash_bytes(&expected));
}

#[test]
fn deepseek_v32_mla_snapshot_matches_runtime_fp8_ds_mla_page() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let hidden = 4usize;
    let heads = 1usize;
    let kv_heads = 1usize;
    let qk_nope = 128usize;
    let qk_rope = 64usize;
    let head_dim = qk_nope + qk_rope;
    let kv_lora = 512usize;
    let v_head = 128usize;
    let intermediate = 4usize;
    let vocab_size = 8usize;
    let mut layer = tiny_deepseek_v32_descriptor_layer();
    let deepseek = layer
        .deepseek
        .as_mut()
        .expect("tiny DeepSeek V3.2 layer should carry DeepSeek metadata");
    deepseek.q_lora_rank = 2;
    deepseek.kv_lora_rank = kv_lora;
    deepseek.qk_nope_head_dim = qk_nope;
    deepseek.qk_rope_head_dim = qk_rope;
    deepseek.v_head_dim = v_head;
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: hidden as u32,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        intermediate: intermediate as u32,
        vocab_size: vocab_size as u32,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept full-size V3.2 MLA cache dimensions");
    assert_eq!(plan.deepseek_kv_cache_width, (kv_lora + qk_rope) as u32);
    assert_ne!(plan.w_k, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.deepseek_kv_a_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.rms_attn, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.k_norm, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    for dim in 0..hidden {
        weight_storage[dim] = f32_to_bf16_bits(1.0);
        write_arena_f32(&mut weight_storage, plan.rms_attn + (dim * 2) as u64, 1.0);
    }
    for dim in 0..kv_lora {
        write_arena_f32(&mut weight_storage, plan.k_norm + (dim * 2) as u64, 1.0);
    }
    let one_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
    for row in 0..(kv_lora + qk_rope) {
        write_arena_byte(&mut weight_storage, plan.w_k, row * hidden, one_fp8);
    }
    let scale_blocks = (kv_lora + qk_rope).div_ceil(128);
    for block in 0..scale_blocks {
        write_arena_f32(
            &mut weight_storage,
            plan.deepseek_kv_a_scale + (block * 2) as u64,
            1.0,
        );
    }

    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16,
        hidden,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        vocab_size,
        max_context_tokens: 4,
        rms_eps: 1.0e-5,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: CudaHfDecodeSequenceExperimentalRtConfig::default(),
    };
    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3.2 packed MLA cache session should create: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V3.2 packed MLA cache session handle should exist");
    let summary = session.run(&[0], 1, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V3.2 one-token run should populate the packed MLA cache: {:?}",
        summary.error
    );

    let page_bytes = 64 * 656;
    let snapshot = session.deepseek_v32_mla_packed_kv_snapshot(0, page_bytes);
    assert_eq!(
        snapshot.status,
        SmokeStatus::Ok,
        "V3.2 packed MLA snapshot should copy device cache: {:?}",
        snapshot.error
    );
    assert_eq!(snapshot.block_count, 1);
    assert_eq!(snapshot.layer_offset_bytes, 0);
    assert_eq!(snapshot.layer_bytes, page_bytes as u64);
    assert_eq!(snapshot.page_bytes, page_bytes as u64);
    assert_eq!(snapshot.copied_bytes, page_bytes as u64);

    let mut expected = vec![0u8; page_bytes];
    let scale = 1.0f32 / 256.0;
    let nope_byte = f32_to_f8_e4m3fn_bits_nearest(256.0);
    expected[..512].fill(nope_byte);
    for scale_index in 0..4usize {
        let offset = 512 + scale_index * core::mem::size_of::<f32>();
        expected[offset..offset + core::mem::size_of::<f32>()]
            .copy_from_slice(&scale.to_le_bytes());
    }
    let one = f32_to_bf16_bits(1.0).to_le_bytes();
    for dim in 0..64usize {
        let offset = 528 + dim * core::mem::size_of::<u16>();
        expected[offset..offset + core::mem::size_of::<u16>()].copy_from_slice(&one);
    }
    assert_page_bytes_eq(
        &snapshot.bytes,
        &expected,
        "V3.2 runtime packed MLA page must match vLLM fp8_ds_mla token-row layout",
    );
    assert_eq!(snapshot.output_hash, fnv_hash_bytes(&expected));
}

#[test]
fn deepseek_v32_indexer_snapshot_matches_vllm_paged_fp8_k_cache_layout() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let hidden = 4usize;
    let heads = 1usize;
    let kv_heads = 1usize;
    let qk_nope = 128usize;
    let qk_rope = 64usize;
    let head_dim = qk_nope + qk_rope;
    let kv_lora = 512usize;
    let v_head = 128usize;
    let index_head_dim = 128usize;
    let intermediate = 4usize;
    let vocab_size = 8usize;
    let mut layer = tiny_deepseek_v32_descriptor_layer();
    let deepseek = layer
        .deepseek
        .as_mut()
        .expect("tiny DeepSeek V3.2 layer should carry DeepSeek metadata");
    deepseek.q_lora_rank = 2;
    deepseek.kv_lora_rank = kv_lora;
    deepseek.qk_nope_head_dim = qk_nope;
    deepseek.qk_rope_head_dim = qk_rope;
    deepseek.v_head_dim = v_head;
    deepseek.index_n_heads = 1;
    deepseek.index_head_dim = index_head_dim;
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: hidden as u32,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        intermediate: intermediate as u32,
        vocab_size: vocab_size as u32,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept full-size V3.2 indexer dimensions");
    assert_ne!(plan.deepseek_indexer_k, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(
        plan.deepseek_indexer_k_scale,
        CUDA_HF_SEQUENCE_MISSING_OFFSET
    );
    assert_ne!(
        plan.deepseek_indexer_k_norm,
        CUDA_HF_SEQUENCE_MISSING_OFFSET
    );
    assert_ne!(
        plan.deepseek_indexer_k_norm_bias,
        CUDA_HF_SEQUENCE_MISSING_OFFSET
    );

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    for dim in 0..hidden {
        weight_storage[dim] = f32_to_bf16_bits(1.0);
        write_arena_f32(&mut weight_storage, plan.rms_attn + (dim * 2) as u64, 1.0);
    }
    for dim in 0..kv_lora {
        write_arena_f32(&mut weight_storage, plan.k_norm + (dim * 2) as u64, 1.0);
    }
    for dim in 0..index_head_dim {
        write_arena_f32(
            &mut weight_storage,
            plan.deepseek_indexer_k_norm + (dim * 2) as u64,
            1.0,
        );
    }
    let one_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
    for row in 0..(kv_lora + qk_rope) {
        write_arena_byte(&mut weight_storage, plan.w_k, row * hidden, one_fp8);
    }
    let kv_scale_blocks = (kv_lora + qk_rope).div_ceil(128);
    for block in 0..kv_scale_blocks {
        write_arena_f32(
            &mut weight_storage,
            plan.deepseek_kv_a_scale + (block * 2) as u64,
            1.0,
        );
    }
    let pos_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
    let neg_fp8 = f32_to_f8_e4m3fn_bits_nearest(-1.0);
    for row in 0..index_head_dim {
        let value = if row % 2 == 0 { pos_fp8 } else { neg_fp8 };
        write_arena_byte(
            &mut weight_storage,
            plan.deepseek_indexer_k,
            row * hidden,
            value,
        );
    }
    write_arena_f32(&mut weight_storage, plan.deepseek_indexer_k_scale, 1.0);

    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16,
        hidden,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        vocab_size,
        max_context_tokens: 4,
        rms_eps: 0.0,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: CudaHfDecodeSequenceExperimentalRtConfig::default(),
    };
    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3.2 indexer cache session should create: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V3.2 indexer cache session handle should exist");
    let summary = session.run(&[0], 1, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V3.2 one-token run should populate the indexer K cache: {:?}",
        summary.error
    );

    let page_bytes = 64 * (index_head_dim + core::mem::size_of::<f32>());
    let snapshot = session.deepseek_v32_indexer_kv_snapshot(0, page_bytes);
    assert_eq!(
        snapshot.status,
        SmokeStatus::Ok,
        "V3.2 indexer KV snapshot should copy device cache: {:?}",
        snapshot.error
    );
    assert_eq!(snapshot.block_count, 1);
    assert_eq!(snapshot.layer_offset_bytes, 0);
    assert_eq!(snapshot.layer_bytes, page_bytes as u64);
    assert_eq!(snapshot.page_bytes, page_bytes as u64);
    assert_eq!(snapshot.copied_bytes, page_bytes as u64);

    let mut expected = vec![0u8; page_bytes];
    let value_bytes = 64 * index_head_dim;
    let scale = 1.0f32 / 256.0;
    let pos_quant = f32_to_f8_e4m3fn_bits_nearest(256.0);
    let neg_quant = f32_to_f8_e4m3fn_bits_nearest(-256.0);
    for dim in 0..index_head_dim {
        let tile_store_offset = (dim / 16) * 16 * 16 + dim % 16;
        expected[tile_store_offset] = if dim % 2 == 0 { pos_quant } else { neg_quant };
    }
    expected[value_bytes..value_bytes + core::mem::size_of::<f32>()]
        .copy_from_slice(&scale.to_le_bytes());
    assert_page_bytes_eq(
        &snapshot.bytes,
        &expected,
        "V3.2 indexer K cache page must match vLLM tiled fp8+f32-scale layout",
    );
    assert_eq!(snapshot.output_hash, fnv_hash_bytes(&expected));
}

#[test]
fn deepseek_v32_indexer_query_state_matches_vllm_quantized_query_and_weights() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let hidden = 4usize;
    let heads = 2usize;
    let kv_heads = 1usize;
    let head_dim = 2usize;
    let intermediate = 4usize;
    let vocab_size = 8usize;
    let mut layer = tiny_deepseek_v32_descriptor_layer();
    let deepseek = layer
        .deepseek
        .as_mut()
        .expect("tiny DeepSeek V3.2 layer should carry DeepSeek metadata");
    deepseek.index_n_heads = 1;
    deepseek.index_head_dim = 4;
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: hidden as u32,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        intermediate: intermediate as u32,
        vocab_size: vocab_size as u32,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V3.2 indexer dimensions");

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    for dim in 0..hidden {
        weight_storage[dim] = f32_to_bf16_bits(1.0);
        write_arena_f32(&mut weight_storage, plan.rms_attn + (dim * 2) as u64, 1.0);
    }
    for dim in 0..2usize {
        write_arena_f32(&mut weight_storage, plan.q_norm + (dim * 2) as u64, 1.0);
        write_arena_f32(&mut weight_storage, plan.k_norm + (dim * 2) as u64, 1.0);
    }

    let one_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
    let neg_one_fp8 = f32_to_f8_e4m3fn_bits_nearest(-1.0);
    let half_fp8 = f32_to_f8_e4m3fn_bits_nearest(0.5);
    let neg_half_fp8 = f32_to_f8_e4m3fn_bits_nearest(-0.5);
    write_arena_byte(&mut weight_storage, plan.w_q, 0, one_fp8);
    write_arena_byte(&mut weight_storage, plan.w_q, hidden, one_fp8);
    write_arena_f32(&mut weight_storage, plan.deepseek_q_a_scale, 1.0);

    for row in 0..3usize {
        write_arena_byte(&mut weight_storage, plan.w_k, row * hidden, one_fp8);
    }
    write_arena_f32(&mut weight_storage, plan.deepseek_kv_a_scale, 1.0);

    let indexer_q_values = [one_fp8, neg_one_fp8, half_fp8, neg_half_fp8];
    for (row, value) in indexer_q_values.into_iter().enumerate() {
        write_arena_byte(&mut weight_storage, plan.deepseek_indexer_q, row * 2, value);
    }
    write_arena_f32(&mut weight_storage, plan.deepseek_indexer_q_scale, 1.0);
    weight_storage[plan.deepseek_indexer_weights as usize] = f32_to_bf16_bits(2.0);

    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16,
        hidden,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        vocab_size,
        max_context_tokens: 4,
        rms_eps: 0.0,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: CudaHfDecodeSequenceExperimentalRtConfig::default(),
    };
    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3.2 query state session should create: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V3.2 query state session handle should exist");
    let summary = session.run(&[0], 1, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V3.2 one-token run should populate indexer query state: {:?}",
        summary.error
    );
    assert_eq!(summary.deepseek_indexer_state_writes, 1);

    let query_bytes = 4usize;
    let q_scale_offset = 4usize;
    let indexer_heads = 1usize;
    let weights_offset = q_scale_offset + indexer_heads * core::mem::size_of::<f32>();
    let token_bytes = weights_offset + indexer_heads * core::mem::size_of::<f32>();
    let snapshot = session.deepseek_v32_indexer_query_state_snapshot(0, token_bytes);
    assert_eq!(
        snapshot.status,
        SmokeStatus::Ok,
        "V3.2 indexer query state snapshot should copy device state: {:?}",
        snapshot.error
    );
    assert_eq!(snapshot.block_count, 4);
    assert_eq!(snapshot.layer_offset_bytes, 0);
    assert_eq!(snapshot.layer_bytes, (token_bytes * 4) as u64);
    assert_eq!(snapshot.page_bytes, token_bytes as u64);
    assert_eq!(snapshot.copied_bytes, token_bytes as u64);

    let mut expected = vec![0u8; token_bytes];
    expected[..query_bytes].copy_from_slice(&[
        f32_to_f8_e4m3fn_bits_nearest(256.0),
        f32_to_f8_e4m3fn_bits_nearest(-256.0),
        f32_to_f8_e4m3fn_bits_nearest(128.0),
        f32_to_f8_e4m3fn_bits_nearest(-128.0),
    ]);
    expected[q_scale_offset..q_scale_offset + 4].copy_from_slice(&(1.0f32 / 256.0).to_le_bytes());
    expected[weights_offset..weights_offset + 4].copy_from_slice(&(1.0f32 / 256.0).to_le_bytes());
    assert_page_bytes_eq(
        &snapshot.bytes,
        &expected,
        "V3.2 indexer query state must match vLLM q_fp8/q_scale/weights layout",
    );
    assert_eq!(snapshot.output_hash, fnv_hash_bytes(&expected));
}

#[test]
fn deepseek_v32_sparse_topk_selects_same_slots_as_vllm_decode_scorer() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let hidden = 4usize;
    let heads = 2usize;
    let kv_heads = 1usize;
    let head_dim = 2usize;
    let intermediate = 4usize;
    let vocab_size = 8usize;
    let mut layer = tiny_deepseek_v32_descriptor_layer();
    let deepseek = layer
        .deepseek
        .as_mut()
        .expect("tiny DeepSeek V3.2 layer should carry DeepSeek metadata");
    deepseek.q_lora_rank = 4;
    deepseek.kv_lora_rank = 4;
    deepseek.index_topk = 1;
    deepseek.index_n_heads = 1;
    deepseek.index_head_dim = 4;
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: hidden as u32,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        intermediate: intermediate as u32,
        vocab_size: vocab_size as u32,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept V3.2 sparse top-k dimensions");

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    for token in 0..3usize {
        weight_storage[token * hidden + token] = f32_to_bf16_bits(1.0);
    }
    for dim in 0..hidden {
        write_arena_f32(&mut weight_storage, plan.rms_attn + (dim * 2) as u64, 1.0);
        write_arena_f32(&mut weight_storage, plan.q_norm + (dim * 2) as u64, 1.0);
        write_arena_f32(
            &mut weight_storage,
            plan.deepseek_indexer_k_norm + (dim * 2) as u64,
            1.0,
        );
    }
    for dim in 0..4usize {
        write_arena_f32(&mut weight_storage, plan.k_norm + (dim * 2) as u64, 1.0);
    }

    let one_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
    for row in 0..4usize {
        write_arena_byte(&mut weight_storage, plan.w_q, row * hidden + row, one_fp8);
    }
    write_arena_f32(&mut weight_storage, plan.deepseek_q_a_scale, 1.0);

    for row in 0..4usize {
        write_arena_byte(&mut weight_storage, plan.w_k, row * hidden + row, one_fp8);
    }
    write_arena_f32(&mut weight_storage, plan.deepseek_kv_a_scale, 1.0);

    for (col, weight) in [1.0f32, 2.0, 3.0, 4.0].into_iter().enumerate() {
        write_arena_byte(
            &mut weight_storage,
            plan.w_v,
            4 + col,
            f32_to_f8_e4m3fn_bits_nearest(weight),
        );
    }
    write_arena_f32(&mut weight_storage, plan.deepseek_kv_b_scale, 1.0);

    for row in 0..4usize {
        write_arena_byte(
            &mut weight_storage,
            plan.deepseek_indexer_q,
            row * 4 + row,
            one_fp8,
        );
        write_arena_byte(
            &mut weight_storage,
            plan.deepseek_indexer_k,
            row * hidden + row,
            one_fp8,
        );
        weight_storage[plan.deepseek_indexer_weights as usize + row] = f32_to_bf16_bits(1.0);
    }
    write_arena_f32(&mut weight_storage, plan.deepseek_indexer_q_scale, 1.0);
    write_arena_f32(&mut weight_storage, plan.deepseek_indexer_k_scale, 1.0);

    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16,
        hidden,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        vocab_size,
        max_context_tokens: 8,
        rms_eps: 0.0,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: CudaHfDecodeSequenceExperimentalRtConfig::default(),
    };
    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3.2 sparse top-k session should create: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V3.2 sparse top-k session handle should exist");
    let summary = session.run(&[0, 1, 2], 1, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V3.2 sparse top-k run should complete: {:?}",
        summary.error
    );
    assert_eq!(summary.deepseek_indexer_state_writes, 3);
    assert_eq!(summary.deepseek_indexer_kv_writes, 3);
    assert_eq!(summary.deepseek_sparse_topk_selections, 3);
    assert_eq!(summary.deepseek_sparse_topk_slots_selected, 3);
    assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 5);
    assert_eq!(
        summary.deepseek_raw_attention_tokens_scanned, 3,
        "V3.2 sparse MLA should scan one selected sparse slot per token"
    );

    let expected_hash = (0..3u64).fold(0u64, |acc, position| {
        let selected_slot = position;
        acc + ((position + 1) * 1_315_423_911u64 ^ 2_654_435_761u64 ^ (selected_slot + 1))
    });
    assert_eq!(
        summary.deepseek_sparse_topk_selection_hash, expected_hash,
        "V3.2 sparse top-k should select slots 0, 1, 2 like the vLLM decode scorer"
    );
    let expected_attention_hash = deepseek_sparse_attention_output_hash_head0(&[2.0, 4.0, 6.0]);
    assert_eq!(
        summary.deepseek_sparse_attention_output_hash, expected_attention_hash,
        "V3.2 sparse MLA attention output should match the selected-token value projection"
    );
}

#[test]
fn deepseek_v32_sparse_mla_output_matches_vllm_flashmla_topk2_reference() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let hidden = 4usize;
    let heads = 2usize;
    let kv_heads = 1usize;
    let head_dim = 2usize;
    let intermediate = 4usize;
    let vocab_size = 8usize;
    let mut layer = tiny_deepseek_v32_descriptor_layer();
    let deepseek = layer
        .deepseek
        .as_mut()
        .expect("tiny DeepSeek V3.2 layer should carry DeepSeek metadata");
    deepseek.q_lora_rank = 4;
    deepseek.kv_lora_rank = 4;
    deepseek.index_topk = 2;
    deepseek.index_n_heads = 1;
    deepseek.index_head_dim = 4;
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: hidden as u32,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        intermediate: intermediate as u32,
        vocab_size: vocab_size as u32,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept V3.2 top-k=2 sparse MLA dimensions");

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    for token in 0..3usize {
        weight_storage[token * hidden + token] = f32_to_bf16_bits(1.0);
    }
    for dim in 0..hidden {
        write_arena_f32(&mut weight_storage, plan.rms_attn + (dim * 2) as u64, 1.0);
        write_arena_f32(&mut weight_storage, plan.q_norm + (dim * 2) as u64, 1.0);
        write_arena_f32(
            &mut weight_storage,
            plan.deepseek_indexer_k_norm + (dim * 2) as u64,
            1.0,
        );
    }
    for dim in 0..4usize {
        write_arena_f32(&mut weight_storage, plan.k_norm + (dim * 2) as u64, 1.0);
    }

    let one_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
    for row in 0..4usize {
        write_arena_byte(&mut weight_storage, plan.w_q, row * hidden + row, one_fp8);
    }
    write_arena_f32(&mut weight_storage, plan.deepseek_q_a_scale, 1.0);

    for row in 0..4usize {
        write_arena_byte(&mut weight_storage, plan.w_k, row * hidden + row, one_fp8);
    }
    write_arena_f32(&mut weight_storage, plan.deepseek_kv_a_scale, 1.0);

    for (col, weight) in [1.0f32, 2.0, 3.0, 4.0].into_iter().enumerate() {
        write_arena_byte(
            &mut weight_storage,
            plan.w_v,
            4 + col,
            f32_to_f8_e4m3fn_bits_nearest(weight),
        );
    }
    write_arena_f32(&mut weight_storage, plan.deepseek_kv_b_scale, 2.0);

    for row in 0..4usize {
        write_arena_byte(
            &mut weight_storage,
            plan.deepseek_indexer_q,
            row * 4 + row,
            one_fp8,
        );
        write_arena_byte(
            &mut weight_storage,
            plan.deepseek_indexer_k,
            row * hidden + row,
            one_fp8,
        );
        weight_storage[plan.deepseek_indexer_weights as usize + row] = f32_to_bf16_bits(1.0);
    }
    write_arena_f32(&mut weight_storage, plan.deepseek_indexer_q_scale, 1.0);
    write_arena_f32(&mut weight_storage, plan.deepseek_indexer_k_scale, 1.0);

    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16,
        hidden,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        vocab_size,
        max_context_tokens: 8,
        rms_eps: 0.0,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: CudaHfDecodeSequenceExperimentalRtConfig::default(),
    };
    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3.2 top-k=2 sparse MLA session should create: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V3.2 top-k=2 sparse MLA session handle should exist");
    let summary = session.run(&[0, 1, 2], 1, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V3.2 top-k=2 sparse MLA run should complete: {:?}",
        summary.error
    );
    assert_eq!(summary.deepseek_indexer_state_writes, 3);
    assert_eq!(summary.deepseek_indexer_kv_writes, 3);
    assert_eq!(summary.deepseek_sparse_topk_selections, 3);
    assert_eq!(summary.deepseek_sparse_topk_slots_selected, 5);
    assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 3);
    assert_eq!(
        summary.deepseek_raw_attention_tokens_scanned, 5,
        "V3.2 top-k=2 sparse MLA should scan the vLLM-selected sparse slots"
    );

    let selected_slots: &[&[u64]] = &[&[0], &[0, 1], &[2, 0]];
    let expected_selection_hash =
        selected_slots
            .iter()
            .enumerate()
            .fold(0u64, |acc, (position, slots)| {
                acc + slots.iter().enumerate().fold(0u64, |inner, (rank, slot)| {
                    inner
                        + (((position as u64) + 1) * 1_315_423_911u64
                            ^ ((rank as u64) + 1) * 2_654_435_761u64
                            ^ (*slot + 1))
                })
            });
    assert_eq!(
        summary.deepseek_sparse_topk_selection_hash, expected_selection_hash,
        "V3.2 sparse top-k should select [0], [0,1], [2,0] like the vLLM decode scorer"
    );

    let expected_attention_hash = deepseek_sparse_attention_output_hash_head0(&[4.0, 6.0, 8.0]);
    assert_eq!(
        summary.deepseek_sparse_attention_output_hash, expected_attention_hash,
        "V3.2 sparse MLA output should consume the packed KV-B projection scale like vLLM/FlashMLA"
    );
}

#[test]
fn deepseek_v3_mla_batched_prefill_writes_prompt_cache_rows() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let created =
        create_tiny_deepseek_mla_cache_session(tiny_deepseek_v3_descriptor_layer(), false, 2);
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3 MLA session should create before batched prefill: {:?}",
        created.summary.error
    );
    let mut session = created.session.expect("V3 MLA session handle should exist");
    let prefill = CudaHfDecodeSequenceLoop::start_session(&mut session, &[0, 1], None);
    assert_eq!(
        prefill.status,
        SmokeStatus::Ok,
        "V3 MLA batched prefill should accept a two-token prompt: {:?}",
        prefill.error
    );
    assert_eq!(prefill.kv_tokens, 2);
    assert_eq!(prefill.graph_replays, 1);

    let snapshot = session.deepseek_v3_mla_kv_snapshot(0, 96);
    assert_eq!(
        snapshot.status,
        SmokeStatus::Ok,
        "V3 MLA prefill KV snapshot should copy device cache: {:?}",
        snapshot.error
    );
    assert_eq!(snapshot.page_bytes, 96);

    let one = f32_to_bf16_bits(1.0).to_le_bytes();
    let mut expected = vec![0u8; 96];
    for token in 0..2usize {
        let row = token * 6;
        expected[row..row + 2].copy_from_slice(&one);
        expected[row + 2..row + 4].copy_from_slice(&one);
        expected[row + 4..row + 6].copy_from_slice(&one);
    }
    assert_page_bytes_eq(
        &snapshot.bytes,
        &expected,
        "V3 MLA batched prefill must commit each prompt token to the vLLM-ordered latent cache",
    );
    assert_eq!(snapshot.output_hash, fnv_hash_bytes(&expected));
}

#[test]
fn deepseek_v32_mla_batched_prefill_uses_f32_norm_cache_rows() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let created =
        create_tiny_deepseek_mla_cache_session(tiny_deepseek_v32_descriptor_layer(), true, 2);
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3.2 MLA session should create before batched prefill: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V3.2 MLA session handle should exist");
    let prefill = CudaHfDecodeSequenceLoop::start_session(&mut session, &[0, 1], None);
    assert_eq!(
        prefill.status,
        SmokeStatus::Ok,
        "V3.2 MLA batched prefill should use f32 DeepSeek norm weights: {:?}",
        prefill.error
    );
    assert_eq!(prefill.kv_tokens, 2);
    assert_eq!(prefill.graph_replays, 1);

    let snapshot = session.deepseek_v3_mla_kv_snapshot(0, 96);
    assert_eq!(
        snapshot.status,
        SmokeStatus::Ok,
        "V3.2 MLA prefill KV snapshot should copy device cache: {:?}",
        snapshot.error
    );

    let one = f32_to_bf16_bits(1.0).to_le_bytes();
    let mut expected = vec![0u8; 96];
    for token in 0..2usize {
        let row = token * 6;
        expected[row..row + 2].copy_from_slice(&one);
        expected[row + 2..row + 4].copy_from_slice(&one);
        expected[row + 4..row + 6].copy_from_slice(&one);
    }
    assert_page_bytes_eq(
        &snapshot.bytes,
        &expected,
        "V3.2 MLA batched prefill must commit prompt rows with f32 norm weights",
    );
    assert_eq!(snapshot.output_hash, fnv_hash_bytes(&expected));
}

#[test]
fn deepseek_v32_sparse_indexer_batched_prefill_populates_prefix_state() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut layer = tiny_deepseek_v32_descriptor_layer();
    let deepseek = layer
        .deepseek
        .as_mut()
        .expect("tiny DeepSeek V3.2 layer should carry DeepSeek metadata");
    deepseek.index_topk = 1;
    let created = create_tiny_deepseek_mla_cache_session(layer, true, 3);
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3.2 sparse-indexer MLA session should create before batched prefill: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V3.2 sparse-indexer MLA session handle should exist");
    let prefill = CudaHfDecodeSequenceLoop::start_session(&mut session, &[0, 1, 2], None);
    assert_eq!(
        prefill.status,
        SmokeStatus::Ok,
        "V3.2 sparse-indexer batched prefill should accept a three-token prompt: {:?}",
        prefill.error
    );
    assert_eq!(prefill.kv_tokens, 3);
    assert_eq!(prefill.graph_replays, 1);
    assert_eq!(prefill.deepseek_indexer_state_writes, 3);
    assert_eq!(prefill.deepseek_indexer_kv_writes, 3);
    assert_eq!(prefill.deepseek_sparse_topk_selections, 1);
    assert_eq!(prefill.deepseek_sparse_topk_slots_selected, 1);
    assert_eq!(prefill.deepseek_sparse_topk_candidates_scored, 3);
}

#[test]
fn deepseek_v4_mla_shape_does_not_reuse_v3_cache_contract() {
    let deepseek = tiny_deepseek_v4_descriptor_layer()
        .deepseek
        .expect("fixture should carry DeepSeek metadata");

    assert!(deepseek.is_v4_mla());
    assert!(!deepseek.is_v3_mla());
    assert_eq!(deepseek.v3_mla_shape(2), None);
}

#[test]
fn declared_weight_descriptors_override_legacy_weight_pointers() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let weights = tiny_descriptor_weights();
    let zero = 0x0000;
    let one = 0x3c00;
    let poisoned_embeddings = [zero; 8];
    let poisoned_rms = [zero; 2];
    let poisoned_matrix = [one; 4];
    let poisoned_lm_head = [zero; 8];
    let poisoned_layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &poisoned_rms,
        rms_mlp_weight: &poisoned_rms,
        w_q: &poisoned_matrix,
        w_q_gate: None,
        w_k: &poisoned_matrix,
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &poisoned_matrix,
        w_o: &poisoned_matrix,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &poisoned_matrix,
        w_up: &poisoned_matrix,
        w_down: &poisoned_matrix,
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: None,
        mlp_kind: 0,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: false,
        attention_kind: crate::decode::hf_chain::layer::CUDA_HF_ATTENTION_FULL,
    };
    let poisoned_layers = [poisoned_layer];
    let weight_blocks = weights.blocks();
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
        embeddings: &poisoned_embeddings,
        layers: &poisoned_layers,
        final_norm_weight: &poisoned_rms,
        lm_head: &poisoned_lm_head,
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
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    }
    .run();

    if summary.status != SmokeStatus::Ok {
        assert_eq!(summary.status, SmokeStatus::Unavailable);
        return;
    }
    assert_eq!(summary.tokens, vec![1, 2, 3, 0]);
    assert_eq!(summary.descriptor_gpu_resident_h2d_bytes, 52);
    assert_eq!(summary.descriptor_gpu_staged_h2d_bytes, 48);
    assert_eq!(summary.planned_weight_descriptor_count, 12);
}

#[test]
fn declared_weight_descriptors_accept_null_legacy_weight_pointers() {
    let _guard = super::cuda_lock::cuda_test_lock();

    assert_raw_descriptor_decode_matches_request(CudaHfDecodeSamplerConfig::greedy());
}

#[test]
fn declared_weight_descriptors_support_temperature_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    assert_raw_descriptor_decode_matches_request(CudaHfDecodeSamplerConfig::vllm_default());
}

#[test]
fn declared_sparse_moe_descriptor_footprint_uses_router_and_experts() {
    let sparse_layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_q_gate: None,
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
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: None,
        mlp_kind: CUDA_HF_MLP_SPARSE_MOE,
        moe_intermediate: 2,
        shared_expert_intermediate: 0,
        num_experts: 3,
        experts_per_token: 2,
        norm_topk_prob: true,
        attention_kind: crate::decode::hf_chain::layer::CUDA_HF_ATTENTION_FULL,
    };
    let layers = [sparse_layer];
    let request = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 1,
        kv_heads: 1,
        head_dim: 4,
        intermediate: 8,
        vocab_size: 8,
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
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: 448,
            gpu_resident_weight_bytes: 448,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: 1,
        }),
        weight_blocks: &[],
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    };

    let footprint = estimate_sequence_footprint(&request).unwrap();

    assert_eq!(footprint.resident_weight_bytes, 448);
    assert_eq!(footprint.layout_bytes, 632);
}

#[test]
fn declared_deepseek_v4_descriptor_footprint_counts_storage_widths_and_hc_blocks() {
    let deepseek_layer = tiny_deepseek_v4_descriptor_layer();
    let layers = [deepseek_layer];
    let request = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        steps: 2,
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
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: 1306,
            gpu_resident_weight_bytes: 1306,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: 1,
        }),
        weight_blocks: &[],
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    };

    let footprint = estimate_sequence_footprint(&request).unwrap();

    assert_eq!(footprint.resident_weight_bytes, 1306);
    assert_eq!(footprint.layout_bytes, 632);
    assert_eq!(footprint.resident_kv_bytes, 2992);
}

#[test]
fn declared_deepseek_v4_descriptor_run_reaches_native_execution_guard() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let deepseek_layer = tiny_deepseek_v4_descriptor_layer();
    let layers = [deepseek_layer];
    let weight_storage = vec![0u16; 1306 / 2];
    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: 1306,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let request = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        steps: 2,
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
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: 1306,
            gpu_resident_weight_bytes: 1306,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    };

    let summary = request.run();
    if summary.status == SmokeStatus::Unavailable {
        return;
    }

    assert_eq!(summary.status, SmokeStatus::Failed);
    assert_eq!(summary.planned_footprint.resident_weight_bytes, 1306);
    assert_eq!(summary.planned_weight_descriptor_count, 1);
    assert_eq!(
        summary.planned_weight_descriptor_hash,
        hash_weight_blocks(&weight_blocks)
    );
    assert!(
        summary
            .error
            .as_deref()
            .is_some_and(|error| error.contains("cuda_error=801")),
        "expected cudaErrorNotSupported guard, got {:?}",
        summary.error
    );
}

#[test]
fn deepseek_v32_layout_plan_names_projection_and_indexer_offsets() {
    let layer = tiny_deepseek_v32_descriptor_layer();
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V3.2 descriptor layer");

    assert_eq!(plan.attention_kind, CUDA_HF_ATTENTION_DEEPSEEK_MLA);
    assert_eq!(plan.deepseek_mode, CUDA_HF_DEEPSEEK_MODE_V32_MLA_INDEXER);
    assert_eq!(plan.deepseek_qk_head_dim, 2);
    assert_eq!(plan.deepseek_q_rows, 4);
    assert_eq!(plan.deepseek_kv_cache_width, 3);
    assert_eq!(plan.deepseek_kv_b_rows, 4);
    assert_eq!(plan.deepseek_value_rows, 2);
    assert_eq!(plan.rms_attn, 40);
    assert_eq!(plan.w_q, 48);
    assert_eq!(plan.deepseek_q_a_scale, 52);
    assert_eq!(plan.q_norm, 54);
    assert_eq!(plan.deepseek_q_b, 58);
    assert_eq!(plan.deepseek_q_b_scale, 62);
    assert_eq!(plan.w_k, 64);
    assert_eq!(plan.deepseek_kv_a_scale, 70);
    assert_eq!(plan.k_norm, 72);
    assert_eq!(plan.w_v, 76);
    assert_eq!(plan.deepseek_kv_b_scale, 80);
    assert_eq!(plan.w_o, 82);
    assert_eq!(plan.deepseek_o_a_scale, 86);
    assert_eq!(plan.deepseek_indexer_q, 88);
    assert_eq!(plan.deepseek_indexer_q_scale, 92);
    assert_eq!(plan.deepseek_indexer_k, 94);
    assert_eq!(plan.deepseek_indexer_k_scale, 98);
    assert_eq!(plan.deepseek_indexer_k_norm, 100);
    assert_eq!(plan.deepseek_indexer_k_norm_bias, 104);
    assert_eq!(plan.deepseek_indexer_weights, 108);
    assert_eq!(plan.rms_mlp, 116);
    assert_ne!(plan.final_norm, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.lm_head, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert!(plan.final_norm > plan.rms_mlp);
    assert!(plan.lm_head > plan.final_norm);
    assert_eq!(plan.deepseek_o_b, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_eq!(
        plan.deepseek_compressor_ape,
        CUDA_HF_SEQUENCE_MISSING_OFFSET
    );
    assert_eq!(plan.layout_bytes, 688);
    assert!(plan.resident_weight_bytes > 0);

    let request = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        steps: 2,
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
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: 1,
        }),
        weight_blocks: &[],
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    };
    let footprint = estimate_sequence_footprint(&request).unwrap();

    assert_eq!(footprint.resident_weight_bytes, plan.resident_weight_bytes);
    assert_eq!(footprint.resident_kv_bytes, 616);
    assert_eq!(footprint.scratch_bytes, 200);
}

#[test]
fn deepseek_v32_dense_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v32_descriptor_layer();
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V3.2 descriptor layer");
    let weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        max_context_tokens: 4,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: Default::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }

    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3.2 DeepSeek should pass session creation: {:?}",
        created.summary.error
    );
    let mut session = created.session.expect("V3.2 session handle should exist");
    assert_eq!(
        created.summary.resident_weight_bytes,
        plan.resident_weight_bytes
    );

    let summary = session.run(&[0], 2, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V3.2 DeepSeek dense path should run through sampling: {:?}",
        summary.error
    );
    assert_eq!(summary.steps, 2);
    assert_eq!(summary.tokens.len(), 2);
    assert_eq!(summary.kv_tokens, 2);
    assert_eq!(summary.graph_replays, 2);
    assert_eq!(summary.deepseek_v3_grouped_router_selections, 0);
    assert_eq!(summary.deepseek_v4_bias_router_selections, 0);
    assert_eq!(summary.deepseek_v4_hash_router_selections, 0);
    assert!(summary.graph_nodes > 0);
}

#[test]
fn deepseek_v32_decode_output_projection_scale_reaches_logits() {
    let _guard = super::cuda_lock::cuda_test_lock();

    fn run_with_output_scale(o_scale: f32) -> Option<Vec<u32>> {
        let hidden = 4usize;
        let heads = 2usize;
        let kv_heads = 1usize;
        let head_dim = 2usize;
        let intermediate = 4usize;
        let vocab_size = 8usize;
        let layer = tiny_deepseek_v32_descriptor_layer();
        let layers = [layer];
        let plan = CudaHfDecodeSequenceLayoutPlanRequest {
            hidden: hidden as u32,
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            head_dim: head_dim as u32,
            intermediate: intermediate as u32,
            vocab_size: vocab_size as u32,
            layers: &layers,
            layer_index: 0,
        }
        .plan()
        .expect("native layout planner should accept tiny V3.2 descriptor layer");
        assert_ne!(plan.final_norm, CUDA_HF_SEQUENCE_MISSING_OFFSET);
        assert_ne!(plan.lm_head, CUDA_HF_SEQUENCE_MISSING_OFFSET);
        assert_ne!(plan.deepseek_o_a_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);

        let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
        weight_storage[1] = f32_to_bf16_bits(1.0);

        for dim in 0..hidden {
            write_arena_f32(&mut weight_storage, plan.rms_attn + (dim * 2) as u64, 1.0);
            write_arena_f32(&mut weight_storage, plan.rms_mlp + (dim * 2) as u64, 1.0);
            write_arena_f32(&mut weight_storage, plan.final_norm + (dim * 2) as u64, 1.0);
        }
        for dim in 0..2usize {
            write_arena_f32(&mut weight_storage, plan.q_norm + (dim * 2) as u64, 1.0);
            write_arena_f32(&mut weight_storage, plan.k_norm + (dim * 2) as u64, 1.0);
        }

        let one_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
        write_arena_f32(&mut weight_storage, plan.deepseek_q_a_scale, 1.0);
        write_arena_f32(&mut weight_storage, plan.deepseek_q_b_scale, 1.0);
        write_arena_f32(&mut weight_storage, plan.deepseek_kv_a_scale, 1.0);
        write_arena_f32(&mut weight_storage, plan.deepseek_kv_b_scale, 1.0);
        write_arena_f32(&mut weight_storage, plan.deepseek_o_a_scale, o_scale);

        write_arena_byte(&mut weight_storage, plan.w_k, 1, one_fp8);
        write_arena_byte(&mut weight_storage, plan.w_k, hidden + 1, one_fp8);
        write_arena_byte(&mut weight_storage, plan.w_v, 2, one_fp8);
        write_arena_byte(&mut weight_storage, plan.w_o, 0, one_fp8);

        weight_storage[plan.lm_head as usize + hidden] = f32_to_bf16_bits(1.0);
        weight_storage[plan.lm_head as usize + 2 * hidden] = f32_to_bf16_bits(-1.0);

        let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
            host_source: weight_storage.as_ptr(),
            source_file: core::ptr::null(),
            source_file_len: 0,
            file_offset_begin: 0,
            block_id: 1,
            block_version: 1,
            offset_bytes: 0,
            bytes: plan.resident_weight_bytes,
            strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
            reserved: 0,
        }];
        let config = CudaHfDecodeSequenceSessionConfig {
            dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16,
            hidden,
            heads,
            kv_heads,
            head_dim,
            intermediate,
            vocab_size,
            max_context_tokens: 4,
            rms_eps: 1e-5,
            rope_theta: Some(10_000.0),
            embeddings: &[],
            layers: &layers,
            final_norm_weight: &[],
            lm_head: &[],
            weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
                blocks: 1,
                gpu_resident_blocks: 1,
                gpu_staged_blocks: 0,
                weight_bytes: plan.resident_weight_bytes,
                gpu_resident_weight_bytes: plan.resident_weight_bytes,
                gpu_staged_weight_bytes: 0,
                descriptor_hash: hash_weight_blocks(&weight_blocks),
            }),
            weight_blocks: &weight_blocks,
            detailed_profile: false,
            experimental_rt: Default::default(),
        };
        let created = config.create();
        if created.summary.status == SmokeStatus::Unavailable {
            return None;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V3.2 projection-scale session should create: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V3.2 projection-scale session handle should exist");
        let summary = session.run(&[0], 1, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "V3.2 projection-scale decode should complete: {:?}",
            summary.error
        );
        Some(summary.tokens)
    }

    let Some(positive_tokens) = run_with_output_scale(1.0) else {
        return;
    };
    let zero_tokens = run_with_output_scale(0.0)
        .expect("CUDA device availability should not change between paired runs");
    assert_eq!(positive_tokens, vec![1]);
    assert_eq!(zero_tokens, vec![0]);
}

#[test]
fn deepseek_v32_projection_scales_reach_sparse_decode_outputs() {
    let _guard = super::cuda_lock::cuda_test_lock();

    #[derive(Clone, Copy)]
    struct ScaleCase {
        q_a: f32,
        kv_a: f32,
        q_b: f32,
    }

    fn run_case(case: ScaleCase) -> Option<CudaHfDecodeSequenceSummary> {
        let hidden = 4usize;
        let heads = 2usize;
        let kv_heads = 1usize;
        let head_dim = 2usize;
        let intermediate = 4usize;
        let vocab_size = 8usize;
        let mut layer = tiny_deepseek_v32_descriptor_layer();
        let deepseek = layer
            .deepseek
            .as_mut()
            .expect("tiny DeepSeek V3.2 layer should carry DeepSeek metadata");
        deepseek.q_lora_rank = 4;
        deepseek.kv_lora_rank = 4;
        deepseek.index_topk = 2;
        deepseek.index_n_heads = 1;
        deepseek.index_head_dim = 4;
        let layers = [layer];
        let plan = CudaHfDecodeSequenceLayoutPlanRequest {
            hidden: hidden as u32,
            heads: heads as u32,
            kv_heads: kv_heads as u32,
            head_dim: head_dim as u32,
            intermediate: intermediate as u32,
            vocab_size: vocab_size as u32,
            layers: &layers,
            layer_index: 0,
        }
        .plan()
        .expect("native layout planner should accept V3.2 projection-scale dimensions");
        assert_ne!(plan.deepseek_q_a_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);
        assert_ne!(plan.deepseek_q_b_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);
        assert_ne!(plan.deepseek_kv_a_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);

        let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
        for token in 0..3usize {
            weight_storage[token * hidden + token] = f32_to_bf16_bits(1.0);
        }
        for dim in 0..hidden {
            write_arena_f32(&mut weight_storage, plan.rms_attn + (dim * 2) as u64, 1.0);
            write_arena_f32(&mut weight_storage, plan.rms_mlp + (dim * 2) as u64, 1.0);
            write_arena_f32(&mut weight_storage, plan.final_norm + (dim * 2) as u64, 1.0);
            write_arena_f32(&mut weight_storage, plan.q_norm + (dim * 2) as u64, 1.0);
            write_arena_f32(
                &mut weight_storage,
                plan.deepseek_indexer_k_norm + (dim * 2) as u64,
                1.0,
            );
        }
        for dim in 0..4usize {
            write_arena_f32(&mut weight_storage, plan.k_norm + (dim * 2) as u64, 1.0);
        }

        let one_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
        for row in 0..4usize {
            write_arena_byte(&mut weight_storage, plan.w_q, row * hidden + row, one_fp8);
            write_arena_byte(&mut weight_storage, plan.w_k, row * hidden + row, one_fp8);
            write_arena_byte(
                &mut weight_storage,
                plan.deepseek_q_b,
                row * 4 + row,
                one_fp8,
            );
            write_arena_byte(
                &mut weight_storage,
                plan.deepseek_indexer_q,
                row * 4 + row,
                one_fp8,
            );
            write_arena_byte(
                &mut weight_storage,
                plan.deepseek_indexer_k,
                row * hidden + row,
                one_fp8,
            );
            weight_storage[plan.deepseek_indexer_weights as usize + row] = f32_to_bf16_bits(1.0);
        }
        write_arena_byte(&mut weight_storage, plan.deepseek_q_b, 2, one_fp8);
        write_arena_f32(&mut weight_storage, plan.deepseek_q_a_scale, case.q_a);
        write_arena_f32(&mut weight_storage, plan.deepseek_kv_a_scale, case.kv_a);
        write_arena_f32(&mut weight_storage, plan.deepseek_q_b_scale, case.q_b);
        write_arena_f32(&mut weight_storage, plan.deepseek_indexer_q_scale, 1.0);
        write_arena_f32(&mut weight_storage, plan.deepseek_indexer_k_scale, 1.0);

        for (col, weight) in [1.0f32, 2.0, 3.0, 4.0].into_iter().enumerate() {
            write_arena_byte(
                &mut weight_storage,
                plan.w_v,
                4 + col,
                f32_to_f8_e4m3fn_bits_nearest(weight),
            );
        }
        for (col, weight) in [-2.0f32, -1.0, 2.0, 1.0].into_iter().enumerate() {
            write_arena_byte(
                &mut weight_storage,
                plan.w_v,
                col,
                f32_to_f8_e4m3fn_bits_nearest(weight),
            );
        }
        write_arena_f32(&mut weight_storage, plan.deepseek_kv_b_scale, 2.0);
        write_arena_byte(&mut weight_storage, plan.w_o, 0, one_fp8);
        write_arena_f32(&mut weight_storage, plan.deepseek_o_a_scale, 1.0);

        weight_storage[plan.lm_head as usize + hidden] = f32_to_bf16_bits(1.0);
        weight_storage[plan.lm_head as usize + 2 * hidden] = f32_to_bf16_bits(-1.0);

        let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
            host_source: weight_storage.as_ptr(),
            source_file: core::ptr::null(),
            source_file_len: 0,
            file_offset_begin: 0,
            block_id: 1,
            block_version: 1,
            offset_bytes: 0,
            bytes: plan.resident_weight_bytes,
            strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
            reserved: 0,
        }];
        let config = CudaHfDecodeSequenceSessionConfig {
            dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16,
            hidden,
            heads,
            kv_heads,
            head_dim,
            intermediate,
            vocab_size,
            max_context_tokens: 8,
            rms_eps: 0.0,
            rope_theta: Some(10_000.0),
            embeddings: &[],
            layers: &layers,
            final_norm_weight: &[],
            lm_head: &[],
            weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
                blocks: 1,
                gpu_resident_blocks: 1,
                gpu_staged_blocks: 0,
                weight_bytes: plan.resident_weight_bytes,
                gpu_resident_weight_bytes: plan.resident_weight_bytes,
                gpu_staged_weight_bytes: 0,
                descriptor_hash: hash_weight_blocks(&weight_blocks),
            }),
            weight_blocks: &weight_blocks,
            detailed_profile: false,
            experimental_rt: CudaHfDecodeSequenceExperimentalRtConfig::default(),
        };
        let created = config.create();
        if created.summary.status == SmokeStatus::Unavailable {
            return None;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V3.2 projection scale session should create: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V3.2 projection scale session handle should exist");
        let summary = session.run(&[0, 1, 2], 2, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "V3.2 projection scale run should complete: {:?}",
            summary.error
        );
        Some(summary)
    }

    let Some(reference) = run_case(ScaleCase {
        q_a: 1.0,
        kv_a: 1.0,
        q_b: 16.0,
    }) else {
        return;
    };
    let q_a_zero = run_case(ScaleCase {
        q_a: 0.0,
        kv_a: 1.0,
        q_b: 1.0,
    })
    .expect("CUDA device availability should not change between paired runs");
    let kv_a_zero = run_case(ScaleCase {
        q_a: 1.0,
        kv_a: 0.0,
        q_b: 1.0,
    })
    .expect("CUDA device availability should not change between paired runs");
    let q_b_zero = run_case(ScaleCase {
        q_a: 1.0,
        kv_a: 1.0,
        q_b: 0.0,
    })
    .expect("CUDA device availability should not change between paired runs");

    assert_ne!(
        reference.deepseek_sparse_topk_selection_hash, q_a_zero.deepseek_sparse_topk_selection_hash,
        "V3.2 q_a projection scale must affect the live sparse-indexer output"
    );
    assert_ne!(
        reference.deepseek_sparse_attention_output_hash,
        kv_a_zero.deepseek_sparse_attention_output_hash,
        "V3.2 kv_a projection scale must affect the live sparse-attention output"
    );
    assert_ne!(
        reference.deepseek_sparse_attention_output_hash,
        q_b_zero.deepseek_sparse_attention_output_hash,
        "V3.2 q_b projection scale must affect the live sparse-attention output"
    );
}

#[test]
fn deepseek_v32_sparse_moe_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut layer = tiny_deepseek_v32_descriptor_layer();
    layer.mlp_kind = CUDA_HF_MLP_SPARSE_MOE;
    layer.moe_intermediate = 4;
    layer.shared_expert_intermediate = 2;
    layer.num_experts = 2;
    layer.experts_per_token = 1;
    layer.norm_topk_prob = true;
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.flags |= CUDA_HF_DEEPSEEK_FLAG_MOE | CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS;
        deepseek.router_num_groups = 1;
        deepseek.router_topk_groups = 1;
        deepseek.routed_scaling_factor = 1.0;
        deepseek
    });
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V3.2 sparse MoE layer");
    assert_ne!(plan.w_router, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.w_expert_gate_up, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.w_expert_down, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        max_context_tokens: 4,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: Default::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }

    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V3.2 DeepSeek sparse MoE should pass session creation: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V3.2 sparse MoE session handle should exist");

    let summary = session.run(&[0], 2, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V3.2 DeepSeek sparse MoE path should run through sampling: {:?}",
        summary.error
    );
    assert_eq!(summary.steps, 2);
    assert_eq!(summary.tokens.len(), 2);
    assert_eq!(summary.kv_tokens, 2);
    assert_eq!(summary.graph_replays, 2);
    assert_eq!(
        summary.deepseek_v3_grouped_router_selections,
        summary.graph_replays
    );
    assert_eq!(summary.deepseek_v4_bias_router_selections, 0);
    assert_eq!(summary.deepseek_v4_hash_router_selections, 0);
    assert!(summary.graph_nodes > 0);
}

#[test]
fn deepseek_v4_swa_dense_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v4_swa_dense_descriptor_layer();
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V4 SWA dense layer");
    assert_ne!(
        plan.deepseek_attention_sink,
        CUDA_HF_SEQUENCE_MISSING_OFFSET
    );
    assert_ne!(plan.deepseek_o_b, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        max_context_tokens: 4,
        rms_eps: 1e-5,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: Default::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }

    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V4 SWA dense DeepSeek should pass session creation: {:?}",
        created.summary.error
    );
    let swa_kv_bytes = created.summary.deepseek_v4_swa_kv_bytes as usize;
    let mut session = created
        .session
        .expect("V4 SWA dense session handle should exist");

    let summary = session.run(&[0], 2, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V4 SWA dense DeepSeek path should run through sampling: {:?}",
        summary.error
    );
    assert_eq!(summary.steps, 2);
    assert_eq!(summary.tokens.len(), 2);
    assert_eq!(summary.kv_tokens, 2);
    assert_eq!(summary.graph_replays, 2);
    assert_eq!(summary.deepseek_v3_grouped_router_selections, 0);
    assert_eq!(summary.deepseek_v4_bias_router_selections, 0);
    assert_eq!(summary.deepseek_v4_hash_router_selections, 0);
    assert!(summary.graph_nodes > 0);

    let snapshot = session.deepseek_v4_swa_kv_snapshot(0, swa_kv_bytes);
    assert_eq!(
        snapshot.status,
        SmokeStatus::Ok,
        "V4 SWA packed KV snapshot should copy device cache: {:?}",
        snapshot.error
    );
    assert_eq!(snapshot.block_count, 1);
    assert_eq!(snapshot.layer_offset_bytes, 0);
    assert_eq!(snapshot.layer_bytes, 576);
    assert_eq!(snapshot.page_bytes, 576);
    assert_eq!(snapshot.copied_bytes, 576);
    let expected = expected_zero_deepseek_v4_swa_fp8_ds_mla_page(2, 1, 1, 576);
    assert_eq!(
        snapshot.bytes, expected,
        "V4 SWA packed fp8_ds_mla page bytes must match vLLM offsets"
    );
    assert_eq!(snapshot.output_hash, fnv_hash_bytes(&expected));
}

#[test]
fn deepseek_v4_swa_dense_snapshot_matches_nonzero_fp8_ds_mla_page() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let hidden = 4usize;
    let heads = 2usize;
    let kv_heads = 1usize;
    let head_dim = 2usize;
    let intermediate = 4usize;
    let vocab_size = 8usize;
    let rms_eps = 1.0e-5f32;
    let layer = tiny_deepseek_v4_swa_dense_descriptor_layer();
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: hidden as u32,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        intermediate: intermediate as u32,
        vocab_size: vocab_size as u32,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V4 SWA dense layer");
    assert_ne!(plan.w_k, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.deepseek_kv_a_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.k_norm, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    for dim in 0..hidden {
        weight_storage[dim] = 0x3c00;
        write_descriptor_u16(
            &mut weight_storage,
            plan.rms_attn + dim as u64,
            hidden,
            vocab_size,
            f32_to_bf16_bits(1.0),
        );
    }
    for dim in 0..head_dim {
        write_descriptor_u16(
            &mut weight_storage,
            plan.k_norm + dim as u64,
            hidden,
            vocab_size,
            f32_to_bf16_bits(1.0),
        );
    }
    let one_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
    write_descriptor_byte(
        &mut weight_storage,
        plan.w_k,
        0,
        hidden,
        vocab_size,
        one_fp8,
    );
    write_descriptor_byte(
        &mut weight_storage,
        plan.w_k,
        hidden,
        hidden,
        vocab_size,
        one_fp8,
    );
    write_descriptor_byte(
        &mut weight_storage,
        plan.deepseek_kv_a_scale,
        0,
        hidden,
        vocab_size,
        encode_e8m0_scale(1.0),
    );

    let weight_blocks = descriptor_weight_blocks(
        &weight_storage,
        hidden,
        vocab_size,
        plan.resident_weight_bytes,
    );
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        vocab_size,
        max_context_tokens: 4,
        rms_eps,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: weight_blocks.len() as u32,
            gpu_resident_blocks: weight_blocks.len() as u32,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: Default::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }

    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V4 SWA dense DeepSeek should pass non-zero descriptor session creation: {:?}",
        created.summary.error
    );
    let swa_kv_bytes = created.summary.deepseek_v4_swa_kv_bytes as usize;
    let mut session = created
        .session
        .expect("V4 SWA dense session handle should exist");

    let summary = session.run(&[0], 2, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V4 SWA dense DeepSeek non-zero path should run through sampling: {:?}",
        summary.error
    );
    assert_eq!(summary.steps, 2);
    assert_eq!(summary.tokens, vec![0, 0]);
    assert_eq!(summary.kv_tokens, 2);

    let snapshot = session.deepseek_v4_swa_kv_snapshot(0, swa_kv_bytes);
    assert_eq!(
        snapshot.status,
        SmokeStatus::Ok,
        "V4 SWA packed KV snapshot should copy non-zero device cache: {:?}",
        snapshot.error
    );
    assert_eq!(snapshot.block_count, 1);
    assert_eq!(snapshot.layer_bytes, 576);
    assert_eq!(snapshot.page_bytes, 576);
    assert_eq!(snapshot.copied_bytes, 576);

    let normalized_k = 1.0_f32 / (1.0_f32 + rms_eps).sqrt();
    let expected =
        expected_deepseek_v4_swa_fp8_ds_mla_page(&vec![vec![normalized_k; 2]; 2], 1, 1, 576);
    let zero_expected = expected_zero_deepseek_v4_swa_fp8_ds_mla_page(2, 1, 1, 576);
    assert_ne!(
        expected, zero_expected,
        "non-zero V4 SWA verifier must exercise fp8/bf16 packed contents"
    );
    assert_eq!(
        snapshot.bytes, expected,
        "V4 SWA packed fp8_ds_mla page bytes must match non-zero vLLM offsets"
    );
    assert_eq!(snapshot.output_hash, fnv_hash_bytes(&expected));
}

#[test]
fn deepseek_v4_swa_dense_snapshot_matches_fullsize_fp8_ds_mla_page() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let hidden = 4usize;
    let heads = 1usize;
    let kv_heads = 1usize;
    let qk_nope = 448usize;
    let qk_rope = 64usize;
    let head_dim = qk_nope + qk_rope;
    let intermediate = 4usize;
    let vocab_size = 8usize;
    let rms_eps = 1.0e-5f32;
    let mut layer = tiny_deepseek_v4_swa_dense_descriptor_layer();
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.qk_nope_head_dim = qk_nope;
        deepseek.qk_rope_head_dim = qk_rope;
        deepseek.v_head_dim = head_dim;
        deepseek
    });
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: hidden as u32,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        intermediate: intermediate as u32,
        vocab_size: vocab_size as u32,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept full-size V4 SWA page shape");
    assert_ne!(plan.w_k, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.deepseek_kv_a_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.k_norm, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    for dim in 0..hidden {
        weight_storage[dim] = 0x3c00;
        write_descriptor_u16(
            &mut weight_storage,
            plan.rms_attn + dim as u64,
            hidden,
            vocab_size,
            f32_to_bf16_bits(1.0),
        );
    }
    for dim in 0..head_dim {
        write_descriptor_u16(
            &mut weight_storage,
            plan.k_norm + dim as u64,
            hidden,
            vocab_size,
            f32_to_bf16_bits(1.0),
        );
    }
    let one_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
    for row in 0..head_dim {
        write_descriptor_byte(
            &mut weight_storage,
            plan.w_k,
            row * hidden,
            hidden,
            vocab_size,
            one_fp8,
        );
    }
    for scale in 0..4 {
        write_descriptor_byte(
            &mut weight_storage,
            plan.deepseek_kv_a_scale,
            scale,
            hidden,
            vocab_size,
            encode_e8m0_scale(1.0),
        );
    }

    let weight_blocks = descriptor_weight_blocks(
        &weight_storage,
        hidden,
        vocab_size,
        plan.resident_weight_bytes,
    );
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        vocab_size,
        max_context_tokens: 4,
        rms_eps,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: weight_blocks.len() as u32,
            gpu_resident_blocks: weight_blocks.len() as u32,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: Default::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }

    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V4 SWA full-size page session should create: {:?}",
        created.summary.error
    );
    let swa_kv_bytes = created.summary.deepseek_v4_swa_kv_bytes as usize;
    assert_eq!(swa_kv_bytes, 37440);
    let mut session = created
        .session
        .expect("V4 SWA full-size page session handle should exist");

    let summary = session.run(&[0], 2, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V4 SWA full-size page path should run through sampling: {:?}",
        summary.error
    );
    assert_eq!(summary.steps, 2);
    assert_eq!(summary.tokens, vec![0, 0]);
    assert_eq!(summary.kv_tokens, 2);

    let snapshot = session.deepseek_v4_swa_kv_snapshot(0, swa_kv_bytes);
    assert_eq!(
        snapshot.status,
        SmokeStatus::Ok,
        "V4 SWA full-size packed KV snapshot should copy device cache: {:?}",
        snapshot.error
    );
    assert_eq!(snapshot.block_count, 1);
    assert_eq!(snapshot.layer_bytes, 37440);
    assert_eq!(snapshot.page_bytes, 37440);
    assert_eq!(snapshot.copied_bytes, 37440);

    let normalized_k = 1.0_f32 / (1.0_f32 + rms_eps).sqrt();
    let expected_tokens = vec![
        fullsize_v4_swa_expected_token(0, normalized_k),
        fullsize_v4_swa_expected_token(1, normalized_k),
    ];
    let expected =
        expected_deepseek_v4_swa_fp8_ds_mla_page(&expected_tokens, qk_nope, qk_rope, 37440);
    let zero_expected = expected_zero_deepseek_v4_swa_fp8_ds_mla_page(2, qk_nope, qk_rope, 37440);
    assert_ne!(
        expected, zero_expected,
        "full-size V4 SWA verifier must exercise fp8, bf16, scales, and padding"
    );
    assert_page_bytes_eq(
        &snapshot.bytes,
        &expected,
        "V4 SWA full-size packed page must match vLLM fp8_ds_mla layout",
    );
    assert_eq!(snapshot.output_hash, fnv_hash_bytes(&expected));
}

fn tiny_deepseek_v4_swa_sparse_moe_layer(hash_router: bool) -> CudaHfDecodeChainLayer<'static> {
    let mut layer = tiny_deepseek_v4_swa_dense_descriptor_layer();
    layer.mlp_kind = CUDA_HF_MLP_SPARSE_MOE;
    layer.moe_intermediate = 4;
    layer.shared_expert_intermediate = 2;
    layer.num_experts = 2;
    layer.experts_per_token = 1;
    layer.norm_topk_prob = true;
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.flags |= CUDA_HF_DEEPSEEK_FLAG_MOE;
        if hash_router {
            deepseek.flags |= CUDA_HF_DEEPSEEK_FLAG_HASH_ROUTER;
        } else {
            deepseek.flags |= CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS;
        }
        deepseek.routed_scaling_factor = 1.0;
        deepseek
    });
    layer
}

fn fill_tiny_v4_swa_sparse_moe_descriptor(
    storage: &mut [u16],
    plan: &CudaHfDecodeSequenceLayoutPlan,
    hash_router: bool,
    expert_payload: bool,
    shared_payload: bool,
) {
    const HIDDEN: usize = 4;
    const VOCAB_SIZE: usize = 8;
    const NUM_EXPERTS: usize = 2;
    const MOE_INTERMEDIATE: usize = 4;
    const SHARED_INTERMEDIATE: usize = 2;
    const HALF_HIDDEN: usize = HIDDEN / 2;
    const HALF_INTERMEDIATE: usize = MOE_INTERMEDIATE / 2;

    for token in 0..VOCAB_SIZE {
        for dim in 0..HIDDEN {
            let value = if token == 0 { (dim + 1) as f32 } else { 0.0 };
            storage[token * HIDDEN + dim] = f32_to_bf16_bits(value);
        }
    }
    for dim in 0..HIDDEN {
        storage[plan.rms_attn as usize + dim] = f32_to_bf16_bits(1.0);
        storage[plan.rms_mlp as usize + dim] = f32_to_bf16_bits(1.0);
    }
    for dim in 0..2usize {
        storage[plan.q_norm as usize + dim] = f32_to_bf16_bits(1.0);
        storage[plan.k_norm as usize + dim] = f32_to_bf16_bits(1.0);
    }
    for offset in [plan.deepseek_hc_attn_scale, plan.deepseek_hc_ffn_scale] {
        if offset != CUDA_HF_SEQUENCE_MISSING_OFFSET {
            write_arena_f32(storage, offset, 0.0);
            write_arena_f32(storage, offset + 2, 0.0);
            write_arena_f32(storage, offset + 4, 0.0);
        }
    }
    if plan.deepseek_hc_head_scale != CUDA_HF_SEQUENCE_MISSING_OFFSET {
        write_arena_f32(storage, plan.deepseek_hc_head_scale, 0.0);
    }

    for dim in 0..HIDDEN {
        storage[plan.w_router as usize + dim] = f32_to_bf16_bits(-1.0);
        storage[plan.w_router as usize + HIDDEN + dim] = f32_to_bf16_bits(1.0);
    }
    let router_metadata = plan.w_router + (NUM_EXPERTS * HIDDEN) as u64;
    if hash_router {
        write_arena_u64(storage, router_metadata, 0, 1);
    } else {
        write_arena_f32(storage, router_metadata, 0.0);
        write_arena_f32(storage, router_metadata + 2, 0.0);
    }

    let expert_gate = plan.w_expert_gate_up;
    let expert_gate_scale =
        expert_gate + rank3_byte_slots(NUM_EXPERTS, MOE_INTERMEDIATE, HALF_HIDDEN);
    let gate_scale_cols = HALF_HIDDEN.div_ceil(16);
    let expert_up =
        expert_gate_scale + rank3_byte_slots(NUM_EXPERTS, MOE_INTERMEDIATE, gate_scale_cols);
    let expert_up_scale = expert_up + rank3_byte_slots(NUM_EXPERTS, MOE_INTERMEDIATE, HALF_HIDDEN);
    let expert_down = plan.w_expert_down;
    let expert_down_scale = expert_down + rank3_byte_slots(NUM_EXPERTS, HIDDEN, HALF_INTERMEDIATE);

    let scale = encode_e8m0_scale(1.0);
    write_arena_mxfp4_rank3_scales(
        storage,
        expert_gate_scale,
        MOE_INTERMEDIATE,
        HALF_HIDDEN,
        1,
        scale,
    );
    write_arena_mxfp4_rank3_scales(
        storage,
        expert_up_scale,
        MOE_INTERMEDIATE,
        HALF_HIDDEN,
        1,
        scale,
    );
    write_arena_mxfp4_rank3_scales(
        storage,
        expert_down_scale,
        HIDDEN,
        HALF_INTERMEDIATE,
        1,
        scale,
    );

    if expert_payload {
        for row in 0..MOE_INTERMEDIATE {
            for col in 0..HIDDEN {
                write_arena_mxfp4_rank3_value(
                    storage,
                    expert_gate,
                    MOE_INTERMEDIATE,
                    HALF_HIDDEN,
                    1,
                    row,
                    col,
                    0x2,
                );
                write_arena_mxfp4_rank3_value(
                    storage,
                    expert_up,
                    MOE_INTERMEDIATE,
                    HALF_HIDDEN,
                    1,
                    row,
                    col,
                    0x2,
                );
            }
        }
        for row in 0..HIDDEN {
            for col in 0..MOE_INTERMEDIATE {
                write_arena_mxfp4_rank3_value(
                    storage,
                    expert_down,
                    HIDDEN,
                    HALF_INTERMEDIATE,
                    1,
                    row,
                    col,
                    0x2,
                );
            }
        }
    }

    let shared_gate_scale = plan.w_shared_expert_gate + byte_slots(SHARED_INTERMEDIATE, HIDDEN);
    let shared_up_scale = plan.w_shared_expert_up + byte_slots(SHARED_INTERMEDIATE, HIDDEN);
    let shared_down_scale = plan.w_shared_expert_down + byte_slots(HIDDEN, SHARED_INTERMEDIATE);
    write_arena_byte(storage, shared_gate_scale, 0, scale);
    write_arena_byte(storage, shared_up_scale, 0, scale);
    write_arena_byte(storage, shared_down_scale, 0, scale);
    if shared_payload {
        let one_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
        for row in 0..SHARED_INTERMEDIATE {
            for col in 0..HIDDEN {
                write_arena_byte(
                    storage,
                    plan.w_shared_expert_gate,
                    row * HIDDEN + col,
                    one_fp8,
                );
                write_arena_byte(
                    storage,
                    plan.w_shared_expert_up,
                    row * HIDDEN + col,
                    one_fp8,
                );
            }
        }
        for row in 0..HIDDEN {
            for col in 0..SHARED_INTERMEDIATE {
                write_arena_byte(
                    storage,
                    plan.w_shared_expert_down,
                    row * SHARED_INTERMEDIATE + col,
                    one_fp8,
                );
            }
        }
    }
}

fn run_tiny_v4_swa_sparse_moe_descriptor(
    hash_router: bool,
    expert_payload: bool,
    shared_payload: bool,
) -> Option<(CudaHfDecodeSequenceSummary, u64)> {
    let layer = tiny_deepseek_v4_swa_sparse_moe_layer(hash_router);
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V4 SWA sparse MoE layer");
    assert_ne!(plan.w_router, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.w_expert_gate_up, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.w_expert_down, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.w_shared_expert_gate, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.w_shared_expert_up, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.w_shared_expert_down, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    fill_tiny_v4_swa_sparse_moe_descriptor(
        &mut weight_storage,
        &plan,
        hash_router,
        expert_payload,
        shared_payload,
    );
    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        max_context_tokens: 4,
        rms_eps: 1e-5,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: Default::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return None;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V4 SWA sparse MoE DeepSeek should pass session creation: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V4 SWA sparse MoE session handle should exist");

    let summary = session.run(&[0], 1, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V4 SWA sparse MoE DeepSeek path should decode one token: {:?}",
        summary.error
    );
    assert_eq!(summary.steps, 1);
    assert_eq!(summary.tokens.len(), 1);
    assert_eq!(summary.kv_tokens, 1);
    assert_eq!(summary.graph_replays, 1);

    let residual = session.deepseek_v4_mhc_snapshot(CUDA_HF_DEEPSEEK_V4_MHC_STATE_RESIDUAL, 0, 32);
    assert_eq!(
        residual.status,
        SmokeStatus::Ok,
        "V4 mHC residual snapshot should copy sparse MoE decode state: {:?}",
        residual.error
    );
    Some((summary, residual.output_hash))
}

#[test]
fn deepseek_v4_swa_sparse_moe_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let Some((zero_summary, zero_hash)) =
        run_tiny_v4_swa_sparse_moe_descriptor(false, false, false)
    else {
        return;
    };
    let Some((summary, nonzero_hash)) = run_tiny_v4_swa_sparse_moe_descriptor(false, true, false)
    else {
        return;
    };
    assert_eq!(summary.deepseek_v3_grouped_router_selections, 0);
    assert_eq!(
        summary.deepseek_v4_bias_router_selections,
        summary.graph_replays
    );
    assert_eq!(summary.deepseek_v4_hash_router_selections, 0);
    assert_eq!(
        zero_summary.deepseek_v4_bias_router_selections,
        zero_summary.graph_replays
    );
    assert_ne!(
        nonzero_hash, zero_hash,
        "non-zero V4 MXFP4 expert weights must change the mHC residual state"
    );
    assert!(summary.graph_nodes > 0);
}

#[test]
fn deepseek_v4_swa_hash_moe_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let Some((zero_summary, zero_hash)) = run_tiny_v4_swa_sparse_moe_descriptor(true, false, false)
    else {
        return;
    };
    let Some((summary, nonzero_hash)) = run_tiny_v4_swa_sparse_moe_descriptor(true, true, false)
    else {
        return;
    };
    assert_eq!(summary.deepseek_v3_grouped_router_selections, 0);
    assert_eq!(summary.deepseek_v4_bias_router_selections, 0);
    assert_eq!(
        summary.deepseek_v4_hash_router_selections,
        summary.graph_replays
    );
    assert_eq!(zero_summary.deepseek_v4_bias_router_selections, 0);
    assert_eq!(
        zero_summary.deepseek_v4_hash_router_selections,
        zero_summary.graph_replays
    );
    assert_ne!(
        nonzero_hash, zero_hash,
        "non-zero V4 hash-routed MXFP4 expert weights must change the mHC residual state"
    );
    assert!(summary.graph_nodes > 0);
}

#[test]
fn deepseek_v4_swa_sparse_moe_shared_expert_changes_state() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let Some((_, zero_hash)) = run_tiny_v4_swa_sparse_moe_descriptor(false, false, false) else {
        return;
    };
    let Some((summary, shared_hash)) = run_tiny_v4_swa_sparse_moe_descriptor(false, false, true)
    else {
        return;
    };
    assert_eq!(
        summary.deepseek_v4_bias_router_selections,
        summary.graph_replays
    );
    assert_ne!(
        shared_hash, zero_hash,
        "non-zero V4 shared expert weights must change the mHC residual state"
    );
}

#[test]
fn deepseek_v4_compressed_dense_short_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut layer = tiny_deepseek_v4_swa_dense_descriptor_layer();
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.mode = CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED;
        deepseek.flags |= CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR;
        deepseek.compress_ratio = 4;
        deepseek
    });
    with_tiny_deepseek_v4_descriptor_session(layer, 8, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed dense DeepSeek should pass session creation: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed dense session handle should exist");

        let summary = session.run(&[0], 2, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "V4 compressed dense short context should run through SWA-only sampling: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 2);
        assert_eq!(summary.tokens.len(), 2);
        assert_eq!(summary.kv_tokens, 2);
        assert_eq!(summary.graph_replays, 2);
        assert!(summary.graph_nodes > 0);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 0);
        assert_eq!(summary.deepseek_indexer_state_writes, 0);
        assert_eq!(summary.deepseek_indexer_kv_writes, 0);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 0);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 0);
        assert_eq!(summary.deepseek_sparse_topk_selections, 0);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 0);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 0);
    });
}

#[test]
fn deepseek_v4_swa_dense_respects_sliding_window_limit() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut rt = CudaHfDecodeSequenceExperimentalRtConfig::default();
    rt.local_window_tokens = 3;
    with_tiny_deepseek_v4_descriptor_session_with_rt(
        tiny_deepseek_v4_swa_dense_descriptor_layer(),
        8,
        rt,
        |created| {
            if created.summary.status == SmokeStatus::Unavailable {
                return;
            }
            assert_eq!(
                created.summary.status,
                SmokeStatus::Ok,
                "V4 SWA session should create before sliding-window accounting: {:?}",
                created.summary.error
            );
            let mut session = created.session.expect("V4 SWA session handle should exist");

            let summary = session.run(&[0], 6, None);
            assert_eq!(
                summary.status,
                SmokeStatus::Ok,
                "V4 SWA decode should use the configured local window: {:?}",
                summary.error
            );
            assert_eq!(summary.steps, 6);
            assert_eq!(summary.kv_tokens, 6);
            assert_eq!(summary.graph_replays, 6);
            assert_eq!(summary.deepseek_raw_attention_tokens_scanned, 15);
            assert_eq!(summary.deepseek_compressed_kv_writes, 0);
            assert_eq!(summary.deepseek_compressed_kv_attention_reads, 0);
            assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 0);
        },
    );
}

#[test]
fn deepseek_v4_compressed_indexer_short_session_runs_through_sampling() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v4_descriptor_layer();
    with_tiny_deepseek_v4_descriptor_session(layer, 8, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer DeepSeek should pass session creation: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed-indexer session handle should exist");

        let summary = session.run(&[0], 2, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer short context should run through SWA-only sampling: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 2);
        assert_eq!(summary.tokens.len(), 2);
        assert_eq!(summary.kv_tokens, 2);
        assert_eq!(summary.graph_replays, 2);
        assert!(summary.graph_nodes > 0);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 0);
        assert_eq!(summary.deepseek_indexer_state_writes, summary.graph_replays);
        assert_eq!(summary.deepseek_indexer_kv_writes, 0);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 0);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 0);
        assert_eq!(
            summary.deepseek_raw_attention_tokens_scanned, 3,
            "V4 compressed-indexer must keep the raw SWA path active before the first compressed block"
        );
        assert_eq!(summary.deepseek_sparse_topk_selections, 0);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 0);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 0);
    });
}

#[test]
fn deepseek_v4_compressed_indexer_writes_first_boundary_cache() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v4_descriptor_layer();
    with_tiny_deepseek_v4_descriptor_session(layer, 8, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer DeepSeek should create before runtime boundary checks: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed-indexer session handle should exist");

        let summary = session.run(&[0], 4, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "context length at compress_ratio should write first compressed cache boundary: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 4);
        assert_eq!(summary.tokens.len(), 4);
        assert_eq!(summary.kv_tokens, 4);
        assert_eq!(summary.graph_replays, 4);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 1);
        assert_eq!(summary.deepseek_indexer_state_writes, summary.graph_replays);
        assert_eq!(summary.deepseek_indexer_kv_writes, 1);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 1);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 1);
        assert_eq!(
            summary.deepseek_raw_attention_tokens_scanned, 10,
            "V4 C4 attention should scan the full SWA window in addition to compressed top-k"
        );
        assert_eq!(
            summary.deepseek_raw_attention_tokens_scanned
                + summary.deepseek_compressed_kv_attention_slots_scanned,
            11,
            "V4 C4 mixed sparse length should match vLLM's SWA + extra compressed top-k contract"
        );
        assert_eq!(summary.deepseek_sparse_topk_selections, 1);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 1);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 0);
        assert_eq!(
            summary.deepseek_sparse_topk_selection_hash,
            deepseek_sparse_topk_selection_hash(&[(3, &[0])]),
            "V4 C4 sparse indexer should select the first compressed slot at the first compression boundary"
        );
    });
}

#[test]
fn deepseek_v4_compressed_snapshot_matches_fp8_ds_mla_page() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v4_descriptor_layer();
    with_tiny_deepseek_v4_descriptor_session(layer, 8, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer DeepSeek should create before compressed KV snapshot: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed-indexer session handle should exist");

        let summary = session.run(&[0], 4, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "compress_ratio boundary should write a compressed KV page: {:?}",
            summary.error
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 1);

        let snapshot = session.deepseek_v4_compressed_kv_snapshot(0, 576);
        assert_eq!(
            snapshot.status,
            SmokeStatus::Ok,
            "V4 compressed packed KV snapshot should copy device cache: {:?}",
            snapshot.error
        );
        assert_eq!(snapshot.block_count, 1);
        assert_eq!(snapshot.layer_offset_bytes, 0);
        assert_eq!(snapshot.layer_bytes, 576);
        assert_eq!(snapshot.page_bytes, 576);
        assert_eq!(snapshot.copied_bytes, 576);

        let expected = expected_zero_deepseek_v4_swa_fp8_ds_mla_page(1, 1, 1, 576);
        assert_page_bytes_eq(
            &snapshot.bytes,
            &expected,
            "V4 compressed packed fp8_ds_mla page bytes must match vLLM offsets",
        );
        assert_eq!(snapshot.output_hash, fnv_hash_bytes(&expected));
    });
}

#[test]
fn deepseek_v4_compressed_snapshot_matches_fullsize_nonzero_fp8_ds_mla_page() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let hidden = 4usize;
    let heads = 1usize;
    let kv_heads = 1usize;
    let qk_nope = 448usize;
    let qk_rope = 64usize;
    let head_dim = qk_nope + qk_rope;
    let intermediate = 4usize;
    let vocab_size = 8usize;
    let rms_eps = 1.0e-5f32;
    let mut layer = tiny_deepseek_v4_swa_dense_descriptor_layer();
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.mode = CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED;
        deepseek.flags = CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR;
        deepseek.qk_nope_head_dim = qk_nope;
        deepseek.qk_rope_head_dim = qk_rope;
        deepseek.v_head_dim = head_dim;
        deepseek.compress_ratio = 4;
        deepseek
    });
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: hidden as u32,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        intermediate: intermediate as u32,
        vocab_size: vocab_size as u32,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept full-size V4 compressed page shape");
    assert_ne!(
        plan.deepseek_compressor_wkv,
        CUDA_HF_SEQUENCE_MISSING_OFFSET
    );
    assert_ne!(
        plan.deepseek_compressor_norm,
        CUDA_HF_SEQUENCE_MISSING_OFFSET
    );

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    for dim in 0..hidden {
        weight_storage[dim] = 0x3c00;
        write_descriptor_u16(
            &mut weight_storage,
            plan.rms_attn + dim as u64,
            hidden,
            vocab_size,
            f32_to_bf16_bits(1.0),
        );
    }
    for dim in 0..head_dim {
        write_descriptor_u16(
            &mut weight_storage,
            plan.deepseek_compressor_norm + dim as u64,
            hidden,
            vocab_size,
            f32_to_bf16_bits(1.0),
        );
    }
    let state_width = 2 * head_dim;
    for row in 0..state_width {
        write_descriptor_u16(
            &mut weight_storage,
            plan.deepseek_compressor_wkv + (row * hidden) as u64,
            hidden,
            vocab_size,
            f32_to_bf16_bits(1.0),
        );
    }

    let weight_blocks = descriptor_weight_blocks(
        &weight_storage,
        hidden,
        vocab_size,
        plan.resident_weight_bytes,
    );
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        vocab_size,
        max_context_tokens: 8,
        rms_eps,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: weight_blocks.len() as u32,
            gpu_resident_blocks: weight_blocks.len() as u32,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: Default::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V4 compressed full-size page session should create: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V4 compressed full-size page session handle should exist");

    let summary = session.run(&[0], 4, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "first compression boundary should write a full-size compressed page: {:?}",
        summary.error
    );
    assert_eq!(summary.steps, 4);
    assert_eq!(summary.deepseek_compressor_state_writes, 4);
    assert_eq!(summary.deepseek_compressed_kv_writes, 1);

    let snapshot = session.deepseek_v4_compressed_kv_snapshot(0, 37440);
    assert_eq!(
        snapshot.status,
        SmokeStatus::Ok,
        "V4 compressed full-size packed KV snapshot should copy device cache: {:?}",
        snapshot.error
    );
    assert_eq!(snapshot.block_count, 1);
    assert_eq!(snapshot.layer_offset_bytes, 0);
    assert_eq!(snapshot.layer_bytes, 37440);
    assert_eq!(snapshot.page_bytes, 37440);
    assert_eq!(snapshot.copied_bytes, 37440);

    let normalized_k = 1.0_f32 / (1.0_f32 + rms_eps).sqrt();
    let expected = expected_deepseek_v4_swa_fp8_ds_mla_page(
        &vec![vec![normalized_k; qk_nope + qk_rope]],
        qk_nope,
        qk_rope,
        37440,
    );
    let zero_expected = expected_zero_deepseek_v4_swa_fp8_ds_mla_page(1, qk_nope, qk_rope, 37440);
    assert_ne!(
        expected, zero_expected,
        "full-size V4 compressed verifier must exercise non-zero fp8, bf16, scales, and padding"
    );
    assert_page_bytes_eq(
        &snapshot.bytes,
        &expected,
        "V4 compressed full-size packed page must match vLLM fp8_ds_mla layout",
    );
    assert_eq!(snapshot.output_hash, fnv_hash_bytes(&expected));
}

#[test]
fn deepseek_v4_c128_compressed_snapshot_matches_fullsize_fp8_ds_mla_page() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let hidden = 4usize;
    let heads = 1usize;
    let kv_heads = 1usize;
    let qk_nope = 448usize;
    let qk_rope = 64usize;
    let head_dim = qk_nope + qk_rope;
    let intermediate = 4usize;
    let vocab_size = 8usize;
    let rms_eps = 1.0e-5f32;
    let mut layer = tiny_deepseek_v4_swa_dense_descriptor_layer();
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.mode = CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED;
        deepseek.flags = CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR;
        deepseek.qk_nope_head_dim = qk_nope;
        deepseek.qk_rope_head_dim = qk_rope;
        deepseek.v_head_dim = head_dim;
        deepseek.compress_ratio = 128;
        deepseek
    });
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: hidden as u32,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        intermediate: intermediate as u32,
        vocab_size: vocab_size as u32,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept full-size V4 C128 compressed page shape");
    assert_ne!(
        plan.deepseek_compressor_wkv,
        CUDA_HF_SEQUENCE_MISSING_OFFSET
    );
    assert_ne!(
        plan.deepseek_compressor_norm,
        CUDA_HF_SEQUENCE_MISSING_OFFSET
    );

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    for dim in 0..hidden {
        weight_storage[dim] = 0x3c00;
        write_descriptor_u16(
            &mut weight_storage,
            plan.rms_attn + dim as u64,
            hidden,
            vocab_size,
            f32_to_bf16_bits(1.0),
        );
    }
    for dim in 0..head_dim {
        write_descriptor_u16(
            &mut weight_storage,
            plan.deepseek_compressor_norm + dim as u64,
            hidden,
            vocab_size,
            f32_to_bf16_bits(1.0),
        );
    }
    for row in 0..head_dim {
        write_descriptor_u16(
            &mut weight_storage,
            plan.deepseek_compressor_wkv + (row * hidden) as u64,
            hidden,
            vocab_size,
            f32_to_bf16_bits(1.0),
        );
    }

    let weight_blocks = descriptor_weight_blocks(
        &weight_storage,
        hidden,
        vocab_size,
        plan.resident_weight_bytes,
    );
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        vocab_size,
        max_context_tokens: 128,
        rms_eps,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: weight_blocks.len() as u32,
            gpu_resident_blocks: weight_blocks.len() as u32,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: Default::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V4 C128 compressed full-size page session should create: {:?}",
        created.summary.error
    );
    let mut session = created
        .session
        .expect("V4 C128 compressed full-size page session handle should exist");

    let summary = session.run(&[0], 128, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "C128 compression boundary should write a full-size compressed page: {:?}",
        summary.error
    );
    assert_eq!(summary.steps, 128);
    assert_eq!(summary.deepseek_compressor_state_writes, 128);
    assert_eq!(summary.deepseek_compressed_kv_writes, 1);

    let snapshot = session.deepseek_v4_compressed_kv_snapshot(0, 1728);
    assert_eq!(
        snapshot.status,
        SmokeStatus::Ok,
        "V4 C128 compressed packed KV snapshot should copy device cache: {:?}",
        snapshot.error
    );
    assert_eq!(snapshot.block_count, 1);
    assert_eq!(snapshot.layer_offset_bytes, 0);
    assert_eq!(snapshot.layer_bytes, 1728);
    assert_eq!(snapshot.page_bytes, 1728);
    assert_eq!(snapshot.copied_bytes, 1728);

    let normalized_k = 1.0_f32 / (1.0_f32 + rms_eps).sqrt();
    let expected = expected_deepseek_v4_fp8_ds_mla_page(
        &vec![vec![normalized_k; qk_nope + qk_rope]],
        qk_nope,
        qk_rope,
        1728,
        2,
    );
    let zero_expected = expected_deepseek_v4_fp8_ds_mla_page(
        &vec![vec![0.0; qk_nope + qk_rope]],
        qk_nope,
        qk_rope,
        1728,
        2,
    );
    assert_ne!(
        expected, zero_expected,
        "C128 compressed verifier must exercise non-zero fp8, bf16, scales, and padding"
    );
    assert_page_bytes_eq(
        &snapshot.bytes,
        &expected,
        "V4 C128 compressed page must match the fp8_ds_mla two-token packed layout",
    );
    assert_eq!(snapshot.output_hash, fnv_hash_bytes(&expected));
}

#[test]
fn deepseek_v4_compressed_indexer_writes_realistic_indexer_cache_width() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut layer = tiny_deepseek_v4_descriptor_layer();
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.index_head_dim = 128;
        deepseek
    });
    with_tiny_deepseek_v4_descriptor_session(layer, 8, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer realistic cache width should create: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed-indexer session handle should exist");

        let summary = session.run(&[0], 4, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "realistic indexer head width should write first compressed indexer cache boundary: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 4);
        assert_eq!(summary.kv_tokens, 4);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 1);
        assert_eq!(summary.deepseek_indexer_state_writes, summary.graph_replays);
        assert_eq!(summary.deepseek_indexer_kv_writes, 1);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 1);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 1);
        assert_eq!(summary.deepseek_sparse_topk_selections, 1);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 1);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 0);
    });
}

#[test]
fn deepseek_v4_compressed_indexer_runs_past_first_boundary_with_compressed_attention() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v4_descriptor_layer();
    with_tiny_deepseek_v4_descriptor_session(layer, 8, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer DeepSeek should create before runtime boundary checks: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed-indexer session handle should exist");

        let summary = session.run(&[0], 5, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "context past compress_ratio should read native compressed cache: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 5);
        assert_eq!(summary.tokens.len(), 5);
        assert_eq!(summary.kv_tokens, 5);
        assert_eq!(summary.graph_replays, 5);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 1);
        assert_eq!(summary.deepseek_indexer_state_writes, summary.graph_replays);
        assert_eq!(summary.deepseek_indexer_kv_writes, 1);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 2);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 2);
        assert_eq!(
            summary.deepseek_raw_attention_tokens_scanned, 15,
            "V4 C4 attention should keep scanning SWA tokens after compressed cache appears"
        );
        assert_eq!(
            summary.deepseek_raw_attention_tokens_scanned
                + summary.deepseek_compressed_kv_attention_slots_scanned,
            17,
            "V4 C4 mixed sparse length should be raw SWA plus compressed top-k slots"
        );
        assert_eq!(summary.deepseek_sparse_topk_selections, 2);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 2);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 0);
        assert_eq!(
            summary.deepseek_sparse_topk_selection_hash,
            deepseek_sparse_topk_selection_hash(&[(3, &[0]), (4, &[0])]),
            "V4 C4 sparse indexer should preserve vLLM slot order for cover-all top-k"
        );
    });
}

#[test]
fn deepseek_v4_compressed_indexer_tracks_compressed_attention_scan_growth() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v4_descriptor_layer();
    with_tiny_deepseek_v4_descriptor_session(layer, 12, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer DeepSeek should create before scan accounting: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed-indexer session handle should exist");

        let summary = session.run(&[0], 8, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "context through the second compression boundary should scan compressed slots: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 8);
        assert_eq!(summary.tokens.len(), 8);
        assert_eq!(summary.kv_tokens, 8);
        assert_eq!(summary.graph_replays, 8);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 2);
        assert_eq!(summary.deepseek_indexer_state_writes, summary.graph_replays);
        assert_eq!(summary.deepseek_indexer_kv_writes, 2);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 5);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 6);
        assert_eq!(
            summary.deepseek_raw_attention_tokens_scanned, 36,
            "V4 C4 attention should preserve the full SWA window while compressed scans grow"
        );
        assert_eq!(
            summary.deepseek_raw_attention_tokens_scanned
                + summary.deepseek_compressed_kv_attention_slots_scanned,
            42,
            "V4 C4 cover-all sparse length should combine SWA tokens and compressed slots"
        );
        assert_eq!(summary.deepseek_sparse_topk_selections, 5);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 6);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 0);
        assert_eq!(
            summary.deepseek_sparse_topk_selection_hash,
            deepseek_sparse_topk_selection_hash(&[
                (3, &[0]),
                (4, &[0]),
                (5, &[0]),
                (6, &[0]),
                (7, &[0, 1]),
            ]),
            "V4 C4 sparse indexer should hash cover-all selected compressed slots in vLLM order"
        );
    });
}

#[test]
fn deepseek_v4_compressed_indexer_limits_attention_to_sparse_topk() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut layer = tiny_deepseek_v4_descriptor_layer();
    layer.deepseek = layer.deepseek.map(|mut deepseek| {
        deepseek.index_topk = 1;
        deepseek.index_head_dim = 128;
        deepseek
    });
    with_tiny_deepseek_v4_descriptor_session(layer, 12, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer sparse top-k session should create: {:?}",
            created.summary.error
        );
        let mut session = created
            .session
            .expect("V4 compressed-indexer session handle should exist");

        let summary = session.run(&[0], 8, None);
        assert_eq!(
            summary.status,
            SmokeStatus::Ok,
            "C4 sparse indexer should cap compressed attention slots: {:?}",
            summary.error
        );
        assert_eq!(summary.steps, 8);
        assert_eq!(summary.kv_tokens, 8);
        assert_eq!(
            summary.deepseek_compressor_state_writes,
            summary.graph_replays
        );
        assert_eq!(summary.deepseek_compressed_kv_writes, 2);
        assert_eq!(summary.deepseek_indexer_state_writes, summary.graph_replays);
        assert_eq!(summary.deepseek_indexer_kv_writes, 2);
        assert_eq!(summary.deepseek_compressed_kv_attention_reads, 5);
        assert_eq!(summary.deepseek_compressed_kv_attention_slots_scanned, 5);
        assert_eq!(
            summary.deepseek_raw_attention_tokens_scanned, 36,
            "V4 C4 top-k limiting should not truncate the local SWA window"
        );
        assert_eq!(
            summary.deepseek_raw_attention_tokens_scanned
                + summary.deepseek_compressed_kv_attention_slots_scanned,
            41,
            "V4 C4 top-k limiting should only cap compressed sparse slots"
        );
        assert_eq!(summary.deepseek_sparse_topk_selections, 5);
        assert_eq!(summary.deepseek_sparse_topk_slots_selected, 5);
        assert_eq!(summary.deepseek_sparse_topk_candidates_scored, 2);
        assert_eq!(
            summary.deepseek_sparse_topk_selection_hash,
            deepseek_sparse_topk_selection_hash(&[
                (3, &[0]),
                (4, &[0]),
                (5, &[0]),
                (6, &[0]),
                (7, &[0]),
            ]),
            "V4 C4 sparse top-k=1 should select the vLLM tie-broken lowest compressed slot"
        );
    });
}

#[test]
fn deepseek_v4_compressed_indexer_session_reserves_compressor_runtime_caches() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let mut swa_kv_bytes = None;
    with_tiny_deepseek_v4_descriptor_session(
        tiny_deepseek_v4_swa_dense_descriptor_layer(),
        8,
        |created| {
            if created.summary.status == SmokeStatus::Unavailable {
                return;
            }
            assert_eq!(
                created.summary.status,
                SmokeStatus::Ok,
                "V4 SWA baseline session should create: {:?}",
                created.summary.error
            );
            assert_eq!(
                created.summary.deepseek_v4_attention_aux_streams, 3,
                "V4 SWA attention sessions should provision aux GEMM streams"
            );
            assert_eq!(
                created.summary.deepseek_v4_attention_events, 4,
                "V4 SWA attention sessions should provision aux stream events"
            );
            assert_eq!(
                created.summary.deepseek_v4_swa_kv_bytes, 576,
                "V4 SWA must reserve one vLLM-aligned fp8_ds_mla page"
            );
            assert_eq!(created.summary.deepseek_mhc_residual_bytes, 256);
            assert_eq!(created.summary.deepseek_mhc_post_mix_bytes, 64);
            assert_eq!(created.summary.deepseek_mhc_comb_mix_bytes, 128);
            swa_kv_bytes = Some(created.summary.resident_kv_bytes);
        },
    );

    let Some(swa_kv_bytes) = swa_kv_bytes else {
        return;
    };

    with_tiny_deepseek_v4_descriptor_session(tiny_deepseek_v4_descriptor_layer(), 8, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 compressed-indexer session should create: {:?}",
            created.summary.error
        );
        assert_eq!(
            created.summary.deepseek_v4_attention_aux_streams, 3,
            "V4 attention sessions should provision the vLLM-style aux GEMM streams"
        );
        assert_eq!(
            created.summary.deepseek_v4_attention_events, 4,
            "V4 attention sessions should provision fan-out/join events for aux streams"
        );
        assert_eq!(
            created.summary.deepseek_v4_swa_kv_bytes, 576,
            "V4 compressed-indexer layers still reserve the local SWA fp8_ds_mla page"
        );
        assert_eq!(created.summary.deepseek_mhc_residual_bytes, 256);
        assert_eq!(created.summary.deepseek_mhc_post_mix_bytes, 64);
        assert_eq!(created.summary.deepseek_mhc_comb_mix_bytes, 128);
        assert!(
            created.summary.resident_kv_bytes > swa_kv_bytes,
            "compressed-indexer V4 must reserve compressor/indexer runtime caches: {} <= {}",
            created.summary.resident_kv_bytes,
            swa_kv_bytes
        );
        assert_eq!(
            created.summary.resident_kv_bytes - swa_kv_bytes,
            512 + 576 + 512 + 576,
            "V4 C4 compressed/indexer runtime caches must reserve vLLM-aligned packed pages"
        );
    });

    let mut c128_layer = tiny_deepseek_v4_descriptor_layer();
    c128_layer.deepseek = c128_layer.deepseek.map(|mut deepseek| {
        deepseek.compress_ratio = 128;
        deepseek
    });
    with_tiny_deepseek_v4_descriptor_session(c128_layer, 256, |created| {
        if created.summary.status == SmokeStatus::Unavailable {
            return;
        }
        assert_eq!(
            created.summary.status,
            SmokeStatus::Ok,
            "V4 C128 compressed-indexer session should create: {:?}",
            created.summary.error
        );
        assert_eq!(
            created.summary.deepseek_v4_attention_aux_streams, 3,
            "V4 C128 attention sessions should provision aux GEMM streams"
        );
        assert_eq!(
            created.summary.deepseek_v4_attention_events, 4,
            "V4 C128 attention sessions should provision aux stream events"
        );
        assert_eq!(
            created.summary.deepseek_v4_swa_kv_bytes, 2304,
            "V4 SWA cache must reserve four 64-token fp8_ds_mla pages for 256 tokens"
        );
        assert_eq!(created.summary.deepseek_mhc_residual_bytes, 8192);
        assert_eq!(created.summary.deepseek_mhc_post_mix_bytes, 2048);
        assert_eq!(created.summary.deepseek_mhc_comb_mix_bytes, 4096);
        let mhc_bytes = created.summary.deepseek_mhc_residual_bytes
            + created.summary.deepseek_mhc_post_mix_bytes
            + created.summary.deepseek_mhc_comb_mix_bytes;
        assert_eq!(
            created.summary.resident_kv_bytes
                - 2048
                - created.summary.deepseek_v4_swa_kv_bytes
                - mhc_bytes,
            4096 + 576 + 4096 + 576,
            "V4 C128 compressed/indexer runtime caches must reserve two-token vLLM-aligned packed pages"
        );
    });
}

#[test]
fn deepseek_v4_mhc_snapshot_publishes_runtime_state() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let layer = tiny_deepseek_v4_swa_dense_descriptor_layer();
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V4 mHC descriptor layer");
    assert_eq!(plan.deepseek_hc_mult, 2);
    assert_ne!(plan.deepseek_hc_attn_base, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.deepseek_hc_attn_fn, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.deepseek_hc_attn_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    for (dim, value) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
        weight_storage[dim] = f32_to_bf16_bits(*value);
        weight_storage[plan.rms_attn as usize + dim] = f32_to_bf16_bits(1.0);
        weight_storage[plan.rms_mlp as usize + dim] = f32_to_bf16_bits(1.0);
    }
    for dim in 0..2usize {
        weight_storage[plan.q_norm as usize + dim] = f32_to_bf16_bits(1.0);
        weight_storage[plan.k_norm as usize + dim] = f32_to_bf16_bits(1.0);
    }
    write_arena_f32(&mut weight_storage, plan.deepseek_hc_attn_scale, 0.0);
    write_arena_f32(&mut weight_storage, plan.deepseek_hc_attn_scale + 2, 0.0);
    write_arena_f32(&mut weight_storage, plan.deepseek_hc_attn_scale + 4, 0.0);

    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        max_context_tokens: 4,
        rms_eps: 1e-5,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: CudaHfDecodeSequenceExperimentalRtConfig::default(),
    };

    let created = config.create();
    if created.summary.status == SmokeStatus::Unavailable {
        return;
    }
    assert_eq!(
        created.summary.status,
        SmokeStatus::Ok,
        "V4 mHC runtime session should create: {:?}",
        created.summary.error
    );
    assert_eq!(created.summary.deepseek_mhc_residual_bytes, 128);
    assert_eq!(created.summary.deepseek_mhc_post_mix_bytes, 32);
    assert_eq!(created.summary.deepseek_mhc_comb_mix_bytes, 64);

    let mut session = created.session.expect("V4 mHC session handle should exist");
    let summary = session.run(&[0], 1, None);
    assert_eq!(
        summary.status,
        SmokeStatus::Ok,
        "V4 mHC runtime session should decode one token: {:?}",
        summary.error
    );

    let residual = session.deepseek_v4_mhc_snapshot(CUDA_HF_DEEPSEEK_V4_MHC_STATE_RESIDUAL, 0, 32);
    assert_eq!(
        residual.status,
        SmokeStatus::Ok,
        "V4 mHC residual snapshot should copy device state: {:?}",
        residual.error
    );
    assert_eq!(residual.token_count, 4);
    assert_eq!(residual.token_offset_bytes, 0);
    assert_eq!(residual.token_bytes, 32);
    assert_eq!(residual.total_bytes, 128);
    assert_eq!(residual.copied_bytes, 32);
    assert_eq!(
        f32_values_from_le_bytes(&residual.bytes),
        vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    );

    let post = session.deepseek_v4_mhc_snapshot(CUDA_HF_DEEPSEEK_V4_MHC_STATE_POST_MIX, 0, 8);
    assert_eq!(
        post.status,
        SmokeStatus::Ok,
        "V4 mHC post snapshot should copy device state: {:?}",
        post.error
    );
    assert_eq!(post.token_bytes, 8);
    let post_values = f32_values_from_le_bytes(&post.bytes);
    assert_eq!(post_values.len(), 2);
    for value in post_values {
        assert!((value - 1.0).abs() < 1.0e-6, "unexpected post mix {value}");
    }

    let comb = session.deepseek_v4_mhc_snapshot(CUDA_HF_DEEPSEEK_V4_MHC_STATE_COMB_MIX, 0, 16);
    assert_eq!(
        comb.status,
        SmokeStatus::Ok,
        "V4 mHC comb snapshot should copy device state: {:?}",
        comb.error
    );
    assert_eq!(comb.token_bytes, 16);
    let comb_values = f32_values_from_le_bytes(&comb.bytes);
    assert_eq!(comb_values.len(), 4);
    assert!(
        comb_values
            .iter()
            .all(|value| value.is_finite() && *value > 0.0),
        "mHC comb mix must contain positive finite Sinkhorn weights: {:?}",
        comb_values
    );
    assert_ne!(residual.output_hash, fnv_hash_bytes(&vec![0u8; 32]));
    assert_ne!(post.output_hash, fnv_hash_bytes(&vec![0u8; 8]));
    assert_ne!(comb.output_hash, fnv_hash_bytes(&vec![0u8; 16]));
}

#[test]
fn deepseek_v4_layout_plan_names_compressor_and_indexer_offsets() {
    let layer = tiny_deepseek_v4_descriptor_layer();
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V4 descriptor layer");

    assert_eq!(plan.attention_kind, CUDA_HF_ATTENTION_DEEPSEEK_MLA);
    assert_eq!(
        plan.deepseek_mode,
        CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER
    );
    assert_eq!(plan.deepseek_hc_mult, 2);
    assert_eq!(plan.deepseek_hc_sinkhorn_iters, 20);
    assert_eq!(plan.deepseek_hc_eps, 1.0e-6);
    assert_eq!(plan.deepseek_hc_post_alpha, 2.0);
    assert_eq!(plan.deepseek_compress_rope_theta, 1_000_000.0);
    assert_eq!(plan.deepseek_swiglu_limit, 10.0);
    assert_eq!(plan.deepseek_qk_head_dim, 0);
    assert_eq!(plan.deepseek_q_rows, 0);
    assert_eq!(plan.deepseek_kv_cache_width, 0);
    assert_eq!(plan.deepseek_kv_b_rows, 0);
    assert_eq!(plan.deepseek_value_rows, 0);
    assert_eq!(plan.deepseek_hc_head_base, 40);
    assert_eq!(plan.deepseek_hc_head_fn, 44);
    assert_eq!(plan.deepseek_hc_head_scale, 76);
    assert_eq!(plan.rms_attn, 78);
    assert_eq!(plan.deepseek_hc_attn_base, 82);
    assert_eq!(plan.deepseek_hc_attn_fn, 98);
    assert_eq!(plan.deepseek_hc_attn_scale, 226);
    assert_eq!(plan.deepseek_hc_ffn_base, 232);
    assert_eq!(plan.deepseek_hc_ffn_fn, 248);
    assert_eq!(plan.deepseek_hc_ffn_scale, 376);
    assert_eq!(plan.deepseek_attention_sink, 382);
    assert_eq!(plan.w_q, 386);
    assert_eq!(plan.deepseek_q_a_scale, 390);
    assert_eq!(plan.deepseek_q_b, 391);
    assert_eq!(plan.deepseek_q_b_scale, 395);
    assert_eq!(plan.q_norm, 396);
    assert_eq!(plan.w_k, 398);
    assert_eq!(plan.deepseek_kv_a_scale, 402);
    assert_eq!(plan.k_norm, 403);
    assert_eq!(plan.w_o, 405);
    assert_eq!(plan.deepseek_o_a_scale, 409);
    assert_eq!(plan.deepseek_o_b, 410);
    assert_eq!(plan.deepseek_o_b_scale, 418);
    assert_eq!(plan.deepseek_compressor_ape, 419);
    assert_eq!(plan.deepseek_compressor_wkv, 451);
    assert_eq!(plan.deepseek_compressor_wgate, 467);
    assert_eq!(plan.deepseek_compressor_norm, 483);
    assert_eq!(plan.deepseek_indexer_q, 485);
    assert_eq!(plan.deepseek_indexer_q_scale, 487);
    assert_eq!(plan.deepseek_indexer_compressor_ape, 488);
    assert_eq!(plan.deepseek_indexer_compressor_wkv, 520);
    assert_eq!(plan.deepseek_indexer_compressor_wgate, 536);
    assert_eq!(plan.deepseek_indexer_compressor_norm, 552);
    assert_eq!(plan.deepseek_indexer_weights, 554);
    assert_eq!(plan.rms_mlp, 558);
    assert_eq!(plan.deepseek_indexer_k, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_eq!(plan.deepseek_kv_b_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_eq!(plan.layout_bytes, 688);
}

fn with_tiny_deepseek_v4_descriptor_session(
    layer: CudaHfDecodeChainLayer<'static>,
    max_context_tokens: usize,
    run: impl FnOnce(
        crate::decode::hf_sequence::session::request::CudaHfDecodeSequenceSessionCreateOutput,
    ),
) {
    with_tiny_deepseek_v4_descriptor_session_with_rt(
        layer,
        max_context_tokens,
        CudaHfDecodeSequenceExperimentalRtConfig::default(),
        run,
    );
}

fn with_tiny_deepseek_v4_descriptor_session_with_rt(
    layer: CudaHfDecodeChainLayer<'static>,
    max_context_tokens: usize,
    experimental_rt: CudaHfDecodeSequenceExperimentalRtConfig,
    run: impl FnOnce(
        crate::decode::hf_sequence::session::request::CudaHfDecodeSequenceSessionCreateOutput,
    ),
) {
    let layers = [layer];
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny V4 descriptor layer");
    assert_ne!(
        plan.deepseek_attention_sink,
        CUDA_HF_SEQUENCE_MISSING_OFFSET
    );
    assert_ne!(plan.deepseek_o_b, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 4,
        vocab_size: 8,
        max_context_tokens,
        rms_eps: 1e-5,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt,
    };

    run(config.create());
}

fn tiny_deepseek_v32_descriptor_layer() -> CudaHfDecodeChainLayer<'static> {
    CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_q_gate: None,
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
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: Some(CudaHfDeepSeekLayer {
            mode: CUDA_HF_DEEPSEEK_MODE_V32_MLA_INDEXER,
            flags: CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER,
            hc_mult: 0,
            hc_sinkhorn_iters: 0,
            q_lora_rank: 2,
            kv_lora_rank: 2,
            o_lora_rank: 0,
            o_groups: 0,
            qk_nope_head_dim: 1,
            qk_rope_head_dim: 1,
            v_head_dim: 1,
            compress_ratio: 1,
            index_topk: 0,
            index_n_heads: 2,
            index_head_dim: 2,
            router_num_groups: 1,
            router_topk_groups: 1,
            routed_scaling_factor: 1.0,
            hc_eps: 0.0,
            hc_post_alpha: 0.0,
            rope_scaling_type: CUDA_HF_DEEPSEEK_ROPE_SCALING_NONE,
            rope_original_max_position: 0,
            rope_scaling_factor: 0.0,
            rope_extrapolation_factor: 1.0,
            rope_attn_factor: 1.0,
            rope_beta_fast: 32.0,
            rope_beta_slow: 1.0,
            rope_mscale: 1.0,
            rope_mscale_all_dim: 0.0,
            compress_rope_theta: None,
            swiglu_limit: None,
        }),
        mlp_kind: 0,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: true,
        attention_kind: CUDA_HF_ATTENTION_DEEPSEEK_MLA,
    }
}

fn tiny_deepseek_v3_descriptor_layer() -> CudaHfDecodeChainLayer<'static> {
    CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_q_gate: None,
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
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: Some(CudaHfDeepSeekLayer {
            mode: CUDA_HF_DEEPSEEK_MODE_V3_MLA,
            flags: 0,
            hc_mult: 0,
            hc_sinkhorn_iters: 0,
            q_lora_rank: 2,
            kv_lora_rank: 2,
            o_lora_rank: 0,
            o_groups: 0,
            qk_nope_head_dim: 1,
            qk_rope_head_dim: 1,
            v_head_dim: 1,
            compress_ratio: 1,
            index_topk: 0,
            index_n_heads: 0,
            index_head_dim: 0,
            router_num_groups: 1,
            router_topk_groups: 1,
            routed_scaling_factor: 1.0,
            hc_eps: 0.0,
            hc_post_alpha: 0.0,
            rope_scaling_type: CUDA_HF_DEEPSEEK_ROPE_SCALING_NONE,
            rope_original_max_position: 0,
            rope_scaling_factor: 0.0,
            rope_extrapolation_factor: 1.0,
            rope_attn_factor: 1.0,
            rope_beta_fast: 32.0,
            rope_beta_slow: 1.0,
            rope_mscale: 1.0,
            rope_mscale_all_dim: 0.0,
            compress_rope_theta: None,
            swiglu_limit: None,
        }),
        mlp_kind: CUDA_HF_MLP_DENSE,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: true,
        attention_kind: CUDA_HF_ATTENTION_DEEPSEEK_MLA,
    }
}

fn create_tiny_deepseek_mla_cache_session(
    layer: CudaHfDecodeChainLayer<'static>,
    f32_norms: bool,
    prompt_tokens: usize,
) -> CudaHfDecodeSequenceSessionCreateOutput {
    let hidden = 4usize;
    let heads = 2usize;
    let kv_heads = 1usize;
    let head_dim = 2usize;
    let intermediate = 4usize;
    let vocab_size = 8usize;
    let layers = [layer];
    let active_sparse_indexer = layers[0]
        .deepseek
        .as_ref()
        .is_some_and(|deepseek| deepseek.index_topk > 0);
    let plan = CudaHfDecodeSequenceLayoutPlanRequest {
        hidden: hidden as u32,
        heads: heads as u32,
        kv_heads: kv_heads as u32,
        head_dim: head_dim as u32,
        intermediate: intermediate as u32,
        vocab_size: vocab_size as u32,
        layers: &layers,
        layer_index: 0,
    }
    .plan()
    .expect("native layout planner should accept tiny DeepSeek MLA descriptor layer");
    assert_eq!(plan.deepseek_kv_cache_width, 3);
    assert_ne!(plan.w_k, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.deepseek_kv_a_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.rms_attn, CUDA_HF_SEQUENCE_MISSING_OFFSET);
    assert_ne!(plan.k_norm, CUDA_HF_SEQUENCE_MISSING_OFFSET);

    let mut weight_storage = vec![0u16; (plan.resident_weight_bytes as usize).div_ceil(2)];
    for token in 0..prompt_tokens {
        for dim in 0..hidden {
            weight_storage[token * hidden + dim] = f32_to_bf16_bits(1.0);
        }
    }
    if f32_norms {
        for dim in 0..hidden {
            write_arena_f32(&mut weight_storage, plan.rms_attn + (dim * 2) as u64, 1.0);
        }
        for dim in 0..2usize {
            write_arena_f32(&mut weight_storage, plan.k_norm + (dim * 2) as u64, 1.0);
        }
        if active_sparse_indexer {
            for dim in 0..2usize {
                write_arena_f32(&mut weight_storage, plan.q_norm + (dim * 2) as u64, 1.0);
                write_arena_f32(
                    &mut weight_storage,
                    plan.deepseek_indexer_k_norm + (dim * 2) as u64,
                    1.0,
                );
            }
        }
    } else {
        for dim in 0..hidden {
            weight_storage[plan.rms_attn as usize + dim] = f32_to_bf16_bits(1.0);
        }
        for dim in 0..2usize {
            weight_storage[plan.k_norm as usize + dim] = f32_to_bf16_bits(1.0);
        }
        if active_sparse_indexer {
            for dim in 0..2usize {
                weight_storage[plan.q_norm as usize + dim] = f32_to_bf16_bits(1.0);
                weight_storage[plan.deepseek_indexer_k_norm as usize + dim] = f32_to_bf16_bits(1.0);
            }
        }
    }
    let one_fp8 = f32_to_f8_e4m3fn_bits_nearest(1.0);
    for row in 0..3usize {
        write_arena_byte(&mut weight_storage, plan.w_k, row * hidden, one_fp8);
    }
    write_arena_f32(&mut weight_storage, plan.deepseek_kv_a_scale, 1.0);
    if active_sparse_indexer {
        assert_ne!(plan.w_q, CUDA_HF_SEQUENCE_MISSING_OFFSET);
        assert_ne!(plan.deepseek_q_a_scale, CUDA_HF_SEQUENCE_MISSING_OFFSET);
        assert_ne!(plan.q_norm, CUDA_HF_SEQUENCE_MISSING_OFFSET);
        assert_ne!(plan.deepseek_indexer_q, CUDA_HF_SEQUENCE_MISSING_OFFSET);
        assert_ne!(
            plan.deepseek_indexer_q_scale,
            CUDA_HF_SEQUENCE_MISSING_OFFSET
        );
        assert_ne!(plan.deepseek_indexer_k, CUDA_HF_SEQUENCE_MISSING_OFFSET);
        assert_ne!(
            plan.deepseek_indexer_k_scale,
            CUDA_HF_SEQUENCE_MISSING_OFFSET
        );
        assert_ne!(
            plan.deepseek_indexer_k_norm,
            CUDA_HF_SEQUENCE_MISSING_OFFSET
        );
        assert_ne!(
            plan.deepseek_indexer_k_norm_bias,
            CUDA_HF_SEQUENCE_MISSING_OFFSET
        );
        assert_ne!(
            plan.deepseek_indexer_weights,
            CUDA_HF_SEQUENCE_MISSING_OFFSET
        );
        for row in 0..2usize {
            write_arena_byte(&mut weight_storage, plan.w_q, row * hidden + row, one_fp8);
            write_arena_byte(
                &mut weight_storage,
                plan.deepseek_indexer_k,
                row * hidden + row,
                one_fp8,
            );
        }
        for row in 0..4usize {
            write_arena_byte(
                &mut weight_storage,
                plan.deepseek_indexer_q,
                row * 2 + (row % 2),
                one_fp8,
            );
            weight_storage[plan.deepseek_indexer_weights as usize + row] = f32_to_bf16_bits(1.0);
        }
        write_arena_f32(&mut weight_storage, plan.deepseek_q_a_scale, 1.0);
        write_arena_f32(&mut weight_storage, plan.deepseek_indexer_q_scale, 1.0);
        write_arena_f32(&mut weight_storage, plan.deepseek_indexer_k_scale, 1.0);
    }

    let weight_blocks = [CudaHfDecodeSequenceWeightBlock {
        host_source: weight_storage.as_ptr(),
        source_file: core::ptr::null(),
        source_file_len: 0,
        file_offset_begin: 0,
        block_id: 1,
        block_version: 1,
        offset_bytes: 0,
        bytes: plan.resident_weight_bytes,
        strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
        reserved: 0,
    }];
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16,
        hidden,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        vocab_size,
        max_context_tokens: 4,
        rms_eps: 1.0e-5,
        rope_theta: Some(10_000.0),
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(CudaHfDecodeSequenceWeightPlan {
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: plan.resident_weight_bytes,
            gpu_resident_weight_bytes: plan.resident_weight_bytes,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: hash_weight_blocks(&weight_blocks),
        }),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
        experimental_rt: CudaHfDecodeSequenceExperimentalRtConfig::default(),
    };
    config.create()
}

fn tiny_deepseek_v4_descriptor_layer() -> CudaHfDecodeChainLayer<'static> {
    CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_q_gate: None,
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
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: Some(CudaHfDeepSeekLayer {
            mode: CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER,
            flags: CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR
                | CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER
                | CUDA_HF_DEEPSEEK_FLAG_MOE
                | CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS,
            hc_mult: 2,
            hc_sinkhorn_iters: 20,
            q_lora_rank: 2,
            kv_lora_rank: 1,
            o_lora_rank: 2,
            o_groups: 2,
            qk_nope_head_dim: 1,
            qk_rope_head_dim: 1,
            v_head_dim: 2,
            compress_ratio: 4,
            index_topk: 4,
            index_n_heads: 1,
            index_head_dim: 2,
            router_num_groups: 0,
            router_topk_groups: 0,
            routed_scaling_factor: 1.0,
            hc_eps: 1.0e-6,
            hc_post_alpha: 2.0,
            rope_scaling_type: CUDA_HF_DEEPSEEK_ROPE_SCALING_DEEPSEEK,
            rope_original_max_position: 4096,
            rope_scaling_factor: 40.0,
            rope_extrapolation_factor: 1.0,
            rope_attn_factor: 1.0,
            rope_beta_fast: 32.0,
            rope_beta_slow: 1.0,
            rope_mscale: 1.0,
            rope_mscale_all_dim: 0.0,
            compress_rope_theta: Some(1_000_000.0),
            swiglu_limit: Some(10.0),
        }),
        mlp_kind: CUDA_HF_MLP_SPARSE_MOE,
        moe_intermediate: 4,
        shared_expert_intermediate: 2,
        num_experts: 2,
        experts_per_token: 1,
        norm_topk_prob: true,
        attention_kind: CUDA_HF_ATTENTION_DEEPSEEK_MLA,
    }
}

fn tiny_deepseek_v4_swa_dense_descriptor_layer() -> CudaHfDecodeChainLayer<'static> {
    CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_q_gate: None,
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
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: Some(CudaHfDeepSeekLayer {
            mode: CUDA_HF_DEEPSEEK_MODE_V4_SWA,
            flags: 0,
            hc_mult: 2,
            hc_sinkhorn_iters: 20,
            q_lora_rank: 2,
            kv_lora_rank: 0,
            o_lora_rank: 2,
            o_groups: 1,
            qk_nope_head_dim: 1,
            qk_rope_head_dim: 1,
            v_head_dim: 2,
            compress_ratio: 1,
            index_topk: 0,
            index_n_heads: 0,
            index_head_dim: 0,
            router_num_groups: 0,
            router_topk_groups: 0,
            routed_scaling_factor: 1.0,
            hc_eps: 1.0e-6,
            hc_post_alpha: 2.0,
            rope_scaling_type: CUDA_HF_DEEPSEEK_ROPE_SCALING_NONE,
            rope_original_max_position: 0,
            rope_scaling_factor: 0.0,
            rope_extrapolation_factor: 1.0,
            rope_attn_factor: 1.0,
            rope_beta_fast: 32.0,
            rope_beta_slow: 1.0,
            rope_mscale: 1.0,
            rope_mscale_all_dim: 0.0,
            compress_rope_theta: None,
            swiglu_limit: Some(10.0),
        }),
        mlp_kind: CUDA_HF_MLP_DENSE,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: false,
        attention_kind: CUDA_HF_ATTENTION_DEEPSEEK_MLA,
    }
}

#[test]
fn query_gate_footprint_counts_optional_projection() {
    let zero = 0x0000;
    let embeddings = vec![zero; 8 * 4];
    let rms = vec![zero; 4];
    let attn = vec![zero; 4 * 4];
    let q_gate = vec![zero; 4 * 4];
    let gate = vec![zero; 8 * 4];
    let down = vec![zero; 4 * 8];
    let lm_head = vec![zero; 8 * 4];
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &attn,
        w_q_gate: Some(&q_gate),
        w_k: &attn,
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &attn,
        w_o: &attn,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &gate,
        w_up: &gate,
        w_down: &down,
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: None,
        mlp_kind: 0,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: false,
        attention_kind: crate::decode::hf_chain::layer::CUDA_HF_ATTENTION_FULL,
    };
    let layers = [layer];
    let request = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 1,
        kv_heads: 1,
        head_dim: 4,
        intermediate: 8,
        vocab_size: 8,
        steps: 4,
        seed_token: 0,
        prompt_tokens: &[0],
        eos_token: None,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &embeddings,
        layers: &layers,
        final_norm_weight: &rms,
        lm_head: &lm_head,
        weight_plan: None,
        weight_blocks: &[],
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    };

    let footprint = estimate_sequence_footprint(&request).unwrap();

    assert_eq!(footprint.resident_weight_bytes, 504);
    assert_eq!(footprint.layout_bytes, 632);
}

#[test]
fn linear_gdn_moe_footprint_counts_state_and_scratch() {
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_q_gate: None,
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
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: Some(CudaHfLinearGdnLayer {
            key_heads: 1,
            value_heads: 1,
            key_head_dim: 2,
            value_head_dim: 3,
            conv_kernel: 4,
            w_conv: &[],
            w_qkv: &[],
            w_z: &[],
            w_b: &[],
            w_a: &[],
            dt_bias: &[],
            a_log: &[],
            norm_weight: &[],
            w_out: &[],
        }),
        deepseek: None,
        mlp_kind: CUDA_HF_MLP_SPARSE_MOE,
        moe_intermediate: 3,
        shared_expert_intermediate: 0,
        num_experts: 2,
        experts_per_token: 1,
        norm_topk_prob: true,
        attention_kind: CUDA_HF_ATTENTION_LINEAR_GDN,
    };
    let layers = [layer];
    let request = CudaHfDecodeSequenceRequest {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden: 4,
        heads: 2,
        kv_heads: 1,
        head_dim: 2,
        intermediate: 8,
        vocab_size: 4,
        steps: 2,
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
            blocks: 1,
            gpu_resident_blocks: 1,
            gpu_staged_blocks: 0,
            weight_bytes: 436,
            gpu_resident_weight_bytes: 436,
            gpu_staged_weight_bytes: 0,
            descriptor_hash: 1,
        }),
        weight_blocks: &[],
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    };

    let footprint = estimate_sequence_footprint(&request).unwrap();

    assert_eq!(footprint.resident_weight_bytes, 436);
    assert_eq!(footprint.layout_bytes, 632);
    assert_eq!(footprint.scratch_bytes, 276);
    assert_eq!(footprint.resident_kv_bytes, 128);
    assert_eq!(footprint.device_arena_bytes, 1688);
}

fn assert_raw_descriptor_decode_matches_request(sampler: CudaHfDecodeSamplerConfig) {
    let expected = run_declared_descriptor_decode(sampler);
    if expected.status != SmokeStatus::Ok {
        assert_eq!(expected.status, SmokeStatus::Unavailable);
        return;
    }

    let Some((out, output_tokens)) = run_null_legacy_descriptor_decode(sampler.to_ffi()) else {
        panic!("raw FFI descriptor decode skipped after request decode succeeded");
    };
    assert_eq!(out.status, 0);
    assert_eq!(
        out.descriptor_gpu_resident_h2d_bytes,
        expected.descriptor_gpu_resident_h2d_bytes
    );
    assert_eq!(
        out.descriptor_gpu_staged_h2d_bytes,
        expected.descriptor_gpu_staged_h2d_bytes
    );
    assert_eq!(out.observed_tokens as usize, expected.tokens.len());
    assert_eq!(out.observed_token_hash, expected.observed_token_hash);
    assert_eq!(
        &output_tokens[..expected.tokens.len()],
        expected.tokens.as_slice()
    );
}

fn run_declared_descriptor_decode(
    sampler: CudaHfDecodeSamplerConfig,
) -> CudaHfDecodeSequenceSummary {
    let weights = tiny_descriptor_weights();
    let weight_blocks = weights.blocks();
    let marker_layer = descriptor_marker_layer();
    let layers = [marker_layer];
    CudaHfDecodeSequenceRequest {
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
        sampler,
    }
    .run()
}

fn descriptor_marker_layer() -> CudaHfDecodeChainLayer<'static> {
    CudaHfDecodeChainLayer {
        rms_attn_weight: &[],
        rms_mlp_weight: &[],
        w_q: &[],
        w_q_gate: None,
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
        w_router: None,
        w_expert_gate_up: None,
        w_expert_down: None,
        w_shared_expert_gate: None,
        w_shared_expert_up: None,
        w_shared_expert_down: None,
        w_shared_expert_router: None,
        linear_gdn: None,
        deepseek: None,
        mlp_kind: 0,
        moe_intermediate: 0,
        shared_expert_intermediate: 0,
        num_experts: 0,
        experts_per_token: 0,
        norm_topk_prob: false,
        attention_kind: crate::decode::hf_chain::layer::CUDA_HF_ATTENTION_FULL,
    }
}
