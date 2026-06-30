use crate::deepseek_kv::c4_indexer_topk::deepseek_c4_indexer_topk;
use crate::deepseek_kv::c128_topk::deepseek_c128_topk_metadata;
use crate::deepseek_kv::compress_cache::{
    CudaDeepSeekCompressNormRopeFp8CacheInput, DEEPSEEK_COMPRESS_SCALE_E8M0,
    DEEPSEEK_COMPRESS_SCALE_MXFP4, deepseek_compress_norm_rope_fp8_cache,
};
use crate::deepseek_kv::pack::deepseek_fp8_ds_mla_pack;
use crate::deepseek_kv::partial_states::deepseek_save_partial_states;
use crate::deepseek_kv::slot_mapping::{
    deepseek_compressed_slot_mapping, deepseek_compressed_slot_mapping_reference,
};
use crate::deepseek_kv::summary::{
    CudaDeepSeekC4IndexerTopkSummary, CudaDeepSeekC128TopkMetadataSummary,
    CudaDeepSeekCompressNormRopeFp8CacheSummary, CudaDeepSeekCompressedSlotMappingSummary,
    CudaDeepSeekKvSummary, CudaDeepSeekSavePartialStatesSummary,
};
use crate::smoke::status::SmokeStatus;

const V4_NOPE_BYTES: usize = 448;
const V4_ROPE_BF16_VALUES: usize = 64;
const V4_SCALE_DIM: usize = 8;
const V4_TOKEN_STRIDE: usize = V4_NOPE_BYTES + V4_ROPE_BF16_VALUES * 2;

pub fn deepseek_kv_smoke() -> CudaDeepSeekKvSummary {
    let block_size = 4u32;
    let token_index = 2u32;
    let nope = (0..V4_NOPE_BYTES)
        .map(|idx| (idx as u8).wrapping_mul(3).wrapping_add(7))
        .collect::<Vec<_>>();
    let rope = (0..V4_ROPE_BF16_VALUES)
        .map(|idx| 0x3f80u16.wrapping_add(idx as u16))
        .collect::<Vec<_>>();
    let scales = [0x7f, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x00];

    let summary = deepseek_fp8_ds_mla_pack(block_size, token_index, &nope, &rope, &scales);
    if summary.status != SmokeStatus::Ok {
        return summary;
    }

    if summary.token_stride as usize != V4_TOKEN_STRIDE
        || summary.scale_dim as usize != V4_SCALE_DIM
        || summary.block_bytes as usize != block_size as usize * (V4_TOKEN_STRIDE + V4_SCALE_DIM)
        || !v4_layout_matches(
            &summary.output,
            block_size as usize,
            token_index as usize,
            &nope,
            &rope,
            &scales,
        )
        || summary.kernel_launches != 1
        || summary.sync_calls != 1
        || summary.hot_path_allocations != 0
    {
        let mut failed = summary;
        failed.status = SmokeStatus::Failed;
        failed.error = Some("DeepSeek fp8_ds_mla KV smoke layout mismatch".to_string());
        return failed;
    }

    summary
}

pub fn deepseek_compressed_slot_mapping_smoke() -> CudaDeepSeekCompressedSlotMappingSummary {
    let query_start_loc = [0, 5, 9];
    let seq_lens = [10, 7];
    let block_table = [
        20, 21, 22, 23, // request 0
        30, 31, 32, 33, // request 1
    ];
    let expected = match deepseek_compressed_slot_mapping_reference(
        &query_start_loc,
        &seq_lens,
        &block_table,
        4,
        4,
        4,
    ) {
        Ok(slots) => slots,
        Err(err) => {
            let mut failed = deepseek_compressed_slot_mapping(
                &query_start_loc,
                &seq_lens,
                &block_table,
                4,
                4,
                4,
            );
            failed.status = SmokeStatus::Failed;
            failed.error = Some(format!(
                "DeepSeek compressed slot mapping reference failed: {err}"
            ));
            return failed;
        }
    };

    let summary =
        deepseek_compressed_slot_mapping(&query_start_loc, &seq_lens, &block_table, 4, 4, 4);
    if summary.status != SmokeStatus::Ok {
        return summary;
    }

    if summary.output_slots != expected
        || summary.valid_slots != 2
        || summary.pad_slots != 7
        || summary.kernel_launches != 1
        || summary.sync_calls != 1
        || summary.hot_path_allocations != 0
    {
        let mut failed = summary;
        failed.status = SmokeStatus::Failed;
        failed.error = Some("DeepSeek compressed slot mapping smoke mismatch".to_string());
        return failed;
    }

    summary
}

pub fn deepseek_c128_topk_metadata_smoke() -> CudaDeepSeekC128TopkMetadataSummary {
    let positions = [127, 255, 383, 511];
    let token_to_req = [0, 1, 0, 1];
    let block_table = [
        40, 41, 42, 43, // request 0
        50, 51, 52, 53, // request 1
    ];
    let slot_mapping = [10, -1, 12, 13];
    let expected_global = [
        80, -1, -1, -1, // decode token 0, valid
        100, 101, -1, -1, // decode token 1, invalid slot but row is populated
    ];
    let expected_decode_lens = [1, 0];
    let expected_prefill = [
        0, 1, 2, -1, // prefill token 0
        0, 1, 2, 3, // prefill token 1
    ];

    let summary = deepseek_c128_topk_metadata(
        &positions,
        2,
        &token_to_req,
        &block_table,
        4,
        &slot_mapping,
        2,
        128,
        4,
    );
    if summary.status != SmokeStatus::Ok {
        return summary;
    }

    if summary.global_decode != expected_global
        || summary.decode_lens != expected_decode_lens
        || summary.prefill_local != expected_prefill
        || summary.valid_decode_tokens != 1
        || summary.decode_entries != 1
        || summary.prefill_entries != 7
        || summary.kernel_launches != 1
        || summary.sync_calls != 1
        || summary.hot_path_allocations != 0
    {
        let mut failed = summary;
        failed.status = SmokeStatus::Failed;
        failed.error = Some("DeepSeek C128A top-k metadata smoke mismatch".to_string());
        return failed;
    }

    summary
}

pub fn deepseek_c4_indexer_topk_smoke() -> CudaDeepSeekC4IndexerTopkSummary {
    let (query, key_cache, weights, context_lens) = c4_indexer_topk_fixture();
    let summary = deepseek_c4_indexer_topk(&query, &key_cache, &weights, &context_lens, 2, 2, 2, 2);
    if summary.status != SmokeStatus::Ok {
        return summary;
    }

    if summary.topk_indices != vec![2, 0, 0, 1]
        || !scores_close(&summary.topk_scores, &[1.5, 1.0, 2.0, 0.5])
        || summary.valid_tokens != 2
        || summary.selected_entries != 4
        || summary.kernel_launches != 2
        || summary.sync_calls != 1
        || summary.hot_path_allocations != 0
    {
        let mut failed = summary;
        failed.status = SmokeStatus::Failed;
        failed.error = Some("DeepSeek C4 indexer top-k smoke mismatch".to_string());
        return failed;
    }

    summary
}

pub fn deepseek_save_partial_states_smoke() -> CudaDeepSeekSavePartialStatesSummary {
    let kv = [
        1.0, 2.0, 3.0, // token 0
        4.0, 5.0, 6.0, // token 1 skipped
        7.0, 8.0, 9.0, // token 2
    ];
    let score = [
        0.1, 0.2, 0.3, // token 0
        0.4, 0.5, 0.6, // token 1 skipped
        0.7, 0.8, 0.9, // token 2
    ];
    let ape = [
        10.0, 20.0, 30.0, // row 0
        40.0, 50.0, 60.0, // row 1
        70.0, 80.0, 90.0, // row 2
        100.0, 110.0, 120.0, // row 3
    ];
    let positions = [5, 6, 7];
    let slot_mapping = [1, -1, 4];
    let summary =
        deepseek_save_partial_states(&kv, &score, &ape, &positions, &slot_mapping, 4, 3, 4, 4, 2);
    if summary.status != SmokeStatus::Ok {
        return summary;
    }

    if !save_partial_state_matches(&summary.state_cache)
        || summary.written_tokens != 2
        || summary.skipped_tokens != 1
        || summary.kernel_launches != 1
        || summary.sync_calls != 1
        || summary.hot_path_allocations != 0
    {
        let mut failed = summary;
        failed.status = SmokeStatus::Failed;
        failed.error = Some("DeepSeek save partial states smoke mismatch".to_string());
        return failed;
    }

    summary
}

pub fn deepseek_compress_norm_rope_fp8_cache_smoke() -> CudaDeepSeekCompressNormRopeFp8CacheSummary
{
    let fixture = compress_cache_fixture(DEEPSEEK_COMPRESS_SCALE_E8M0);
    let summary = deepseek_compress_norm_rope_fp8_cache(fixture.input.clone());
    if summary.status != SmokeStatus::Ok {
        return summary;
    }

    let expected = reference_compress_norm_rope_fp8_cache(&fixture);
    if summary.kv_cache == expected
        && summary.written_tokens == 2
        && summary.skipped_tokens == 0
        && summary.kernel_launches == 1
        && summary.sync_calls == 1
        && summary.hot_path_allocations == 0
        && summary.output_hash != 0
    {
        return summary;
    }

    let mut failed = summary;
    failed.status = SmokeStatus::Failed;
    failed.error = Some("DeepSeek fused compress/norm/RoPE FP8 cache smoke mismatch".to_string());
    failed
}

pub fn deepseek_compress_norm_rope_mxfp4_cache_smoke() -> CudaDeepSeekCompressNormRopeFp8CacheSummary
{
    let fixture = mxfp4_compress_cache_fixture();
    let summary = deepseek_compress_norm_rope_fp8_cache(fixture.input.clone());
    if summary.status != SmokeStatus::Ok {
        return summary;
    }

    let expected = reference_compress_norm_rope_fp8_cache(&fixture);
    if summary.kv_cache == expected
        && summary.written_tokens == 2
        && summary.skipped_tokens == 0
        && summary.kernel_launches == 1
        && summary.sync_calls == 1
        && summary.hot_path_allocations == 0
        && summary.output_hash != 0
    {
        return summary;
    }

    let mut failed = summary;
    failed.status = SmokeStatus::Failed;
    failed.error = Some("DeepSeek fused compress/norm/RoPE MXFP4 cache smoke mismatch".to_string());
    failed
}

#[derive(Clone)]
pub(crate) struct CompressCacheFixture {
    pub(crate) input: CudaDeepSeekCompressNormRopeFp8CacheInput<'static>,
}

pub(crate) fn compress_cache_fixture(scale_format: u32) -> CompressCacheFixture {
    let head_size = 4;
    let rope_head_dim = 2;
    let quant_block = 2;
    let token_stride = if scale_format == DEEPSEEK_COMPRESS_SCALE_E8M0 {
        6
    } else {
        4
    };
    let scale_dim = if scale_format == DEEPSEEK_COMPRESS_SCALE_E8M0 {
        2
    } else {
        size_of::<f32>() as u32
    };
    let kv_cache_block_size = 4;
    let kv_cache_block_stride = kv_cache_block_size * (token_stride + scale_dim);
    CompressCacheFixture {
        input: CudaDeepSeekCompressNormRopeFp8CacheInput {
            state_cache: &[
                0.2, -0.3, 0.4, -0.5, 0.0, 0.2, -0.1, 0.3, // row 0
                0.6, -0.7, 0.8, -0.9, 0.4, -0.2, 0.1, -0.5, // row 1
                -1.0, 1.1, -1.2, 1.3, 0.3, 0.6, -0.4, 0.2, // row 2
                1.4, -1.5, 1.6, -1.7, -0.3, 0.7, 0.5, -0.6, // row 3
            ],
            token_to_req_indices: &[0, 0],
            positions: &[1, 3],
            slot_mapping: &[0, 1],
            block_table: &[0],
            kv_slot_mapping: &[0, 1],
            rms_norm_weight: &[1.0, 1.1, 0.9, 1.2],
            cos_sin_cache: &[1.0, 0.0, 0.8, 0.2, 0.6, 0.8],
            num_reqs: 1,
            block_table_stride: 1,
            state_block_size: 4,
            kv_cache_block_size,
            head_size,
            state_width: 4,
            rope_head_dim,
            compress_ratio: 2,
            overlap: 0,
            quant_block,
            token_stride,
            scale_dim,
            scale_format,
            num_state_blocks: 1,
            num_kv_blocks: 1,
            kv_cache_block_stride,
            cos_sin_stride: 2,
            rms_norm_eps: 1.0e-5,
            fp8_max: 448.0,
        },
    }
}

pub(crate) fn mxfp4_compress_cache_fixture() -> CompressCacheFixture {
    let head_size = 128;
    let rope_head_dim = 64;
    let quant_block = 32;
    let token_stride = head_size / 2;
    let scale_dim = head_size / quant_block;
    let kv_cache_block_size = 4;
    let kv_cache_block_stride = kv_cache_block_size * (token_stride + scale_dim);
    let state_cache = leak_f32(
        (0..(4 * head_size * 2))
            .map(|idx| {
                let row = idx / (head_size * 2);
                let dim = idx % (head_size * 2);
                let base = ((idx * 17 + row * 11 + dim * 5) % 127) as f32 / 19.0 - 3.1;
                if dim < head_size {
                    base
                } else {
                    base * 0.31 + row as f32 * 0.17
                }
            })
            .collect(),
    );
    let rms_norm_weight = leak_f32(
        (0..head_size)
            .map(|idx| 0.75 + (idx % 13) as f32 * 0.031)
            .collect(),
    );
    let mut cos_sin = vec![0.0f32; 3 * rope_head_dim];
    for pos in 0..3 {
        for pair in 0..(rope_head_dim / 2) {
            let angle = (pos as f32 + 1.0) * (pair as f32 + 1.0) * 0.003;
            cos_sin[pos * rope_head_dim + pair] = angle.cos();
            cos_sin[pos * rope_head_dim + rope_head_dim / 2 + pair] = angle.sin();
        }
    }
    let cos_sin_cache = leak_f32(cos_sin);
    CompressCacheFixture {
        input: CudaDeepSeekCompressNormRopeFp8CacheInput {
            state_cache,
            token_to_req_indices: &[0, 0],
            positions: &[1, 3],
            slot_mapping: &[0, 1],
            block_table: &[0],
            kv_slot_mapping: &[0, 1],
            rms_norm_weight,
            cos_sin_cache,
            num_reqs: 1,
            block_table_stride: 1,
            state_block_size: 4,
            kv_cache_block_size,
            head_size: head_size as u32,
            state_width: head_size as u32,
            rope_head_dim: rope_head_dim as u32,
            compress_ratio: 2,
            overlap: 0,
            quant_block: quant_block as u32,
            token_stride: token_stride as u32,
            scale_dim: scale_dim as u32,
            scale_format: DEEPSEEK_COMPRESS_SCALE_MXFP4,
            num_state_blocks: 1,
            num_kv_blocks: 1,
            kv_cache_block_stride: kv_cache_block_stride as u32,
            cos_sin_stride: rope_head_dim as u32,
            rms_norm_eps: 1.0e-5,
            fp8_max: 448.0,
        },
    }
}

fn leak_f32(values: Vec<f32>) -> &'static [f32] {
    Box::leak(values.into_boxed_slice())
}

pub(crate) fn reference_compress_norm_rope_fp8_cache(fixture: &CompressCacheFixture) -> Vec<u8> {
    let input = &fixture.input;
    let mut kv_cache =
        vec![0u8; input.num_kv_blocks as usize * input.kv_cache_block_stride as usize];
    for token_idx in 0..input.positions.len() {
        let position = input.positions[token_idx];
        if input.slot_mapping[token_idx] < 0
            || position < 0
            || (position + 1) % input.compress_ratio as i64 != 0
            || input.kv_slot_mapping[token_idx] < 0
        {
            continue;
        }
        let compressed = reference_compressed_state(input, token_idx);
        let variance =
            compressed.iter().map(|value| value * value).sum::<f32>() / input.head_size as f32;
        let rrms = 1.0 / (variance + input.rms_norm_eps).sqrt();
        let normed = compressed
            .iter()
            .zip(input.rms_norm_weight.iter())
            .map(|(value, weight)| value * rrms * weight)
            .collect::<Vec<_>>();
        let rotated = reference_rope(input, position, &normed);
        let kv_slot = input.kv_slot_mapping[token_idx] as usize;
        let kv_pos = kv_slot % input.kv_cache_block_size as usize;
        let data_base = kv_pos * input.token_stride as usize;
        let scale_base = input.kv_cache_block_size as usize * input.token_stride as usize
            + kv_pos * input.scale_dim as usize;
        if input.scale_format == DEEPSEEK_COMPRESS_SCALE_E8M0 {
            let nope = (input.head_size - input.rope_head_dim) as usize;
            for block in 0..(input.head_size / input.quant_block) as usize {
                let start = block * input.quant_block as usize;
                let end = start + input.quant_block as usize;
                let absmax = normed[start..end]
                    .iter()
                    .copied()
                    .map(|value| bf16_to_f32(f32_to_bf16_bits(value)).abs())
                    .fold(0.0f32, f32::max);
                let scale = 2.0f32.powf(((absmax.max(1.0e-4) / input.fp8_max).log2()).ceil());
                if block < nope / input.quant_block as usize {
                    kv_cache[scale_base + block] = encode_e8m0_scale(scale);
                }
                for dim in start..end.min(nope) {
                    let quant_input = bf16_to_f32(f32_to_bf16_bits(normed[dim]));
                    let scaled = (quant_input / scale).clamp(-input.fp8_max, input.fp8_max);
                    kv_cache[data_base + dim] = f32_to_f8_e4m3fn_bits_nearest(scaled);
                }
            }
            kv_cache[scale_base + nope / input.quant_block as usize] = 0;
            for dim in nope..input.head_size as usize {
                let bits = f32_to_bf16_bits(rotated[dim]);
                let offset = data_base + nope + (dim - nope) * 2;
                kv_cache[offset] = (bits & 0xff) as u8;
                kv_cache[offset + 1] = (bits >> 8) as u8;
            }
        } else if input.scale_format
            == crate::deepseek_kv::compress_cache::DEEPSEEK_COMPRESS_SCALE_F32
        {
            let bf16_rotated = rotated
                .iter()
                .copied()
                .map(|value| bf16_to_f32(f32_to_bf16_bits(value)))
                .collect::<Vec<_>>();
            let absmax = bf16_rotated
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0f32, f32::max);
            let scale = 2.0f32.powf(((absmax.max(1.0e-4) / input.fp8_max).log2()).ceil());
            kv_cache[scale_base..scale_base + size_of::<f32>()]
                .copy_from_slice(&scale.to_ne_bytes());
            for (dim, value) in bf16_rotated.iter().copied().enumerate() {
                let scaled = (value / scale).clamp(-input.fp8_max, input.fp8_max);
                kv_cache[data_base + dim] = f32_to_f8_e4m3fn_bits_nearest(scaled);
            }
        } else {
            let half_block = input.quant_block as usize / 2;
            for block in 0..(input.head_size / input.quant_block) as usize {
                let pair_start = block * half_block;
                let mut amax = 0.0f32;
                for pair in 0..half_block {
                    let base = (pair_start + pair) * 2;
                    let even = bf16_to_f32(f32_to_bf16_bits(rotated[base]));
                    let odd = bf16_to_f32(f32_to_bf16_bits(rotated[base + 1]));
                    amax = amax.max(even.abs()).max(odd.abs());
                }
                let exponent =
                    ((amax.max(6.0 * f32::MIN_POSITIVE) / 6.0).log2().ceil()).clamp(-127.0, 127.0);
                let inv_scale = 2.0f32.powf(-exponent);
                kv_cache[scale_base + block] = (exponent as i32 + 127) as u8;
                for pair in 0..half_block {
                    let base = (pair_start + pair) * 2;
                    let even = bf16_to_f32(f32_to_bf16_bits(rotated[base]));
                    let odd = bf16_to_f32(f32_to_bf16_bits(rotated[base + 1]));
                    let lo = f32_to_mxfp4_e2m1_nibble_nearest(even * inv_scale);
                    let hi = f32_to_mxfp4_e2m1_nibble_nearest(odd * inv_scale);
                    kv_cache[data_base + block * half_block + pair] = (hi << 4) | (lo & 0x0f);
                }
            }
        }
    }
    kv_cache
}

fn save_partial_state_matches(state_cache: &[f32]) -> bool {
    let mut expected = vec![0.0f32; 2 * 4 * 2 * 4];
    let row_stride = 8usize;
    let block_stride = 32usize;
    let token0_base = row_stride;
    expected[token0_base..token0_base + 3].copy_from_slice(&[1.0, 2.0, 3.0]);
    expected[token0_base + 4..token0_base + 7].copy_from_slice(&[40.1, 50.2, 60.3]);
    let token2_base = block_stride;
    expected[token2_base..token2_base + 3].copy_from_slice(&[7.0, 8.0, 9.0]);
    expected[token2_base + 4..token2_base + 7].copy_from_slice(&[100.7, 110.8, 120.9]);
    state_cache.len() == expected.len()
        && state_cache
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| (*actual - *expected).abs() < 1e-4)
}

pub fn c4_indexer_topk_fixture() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<i32>) {
    let query = vec![
        1.0, 0.0, // token 0, head 0
        0.0, 1.0, // token 0, head 1
        0.0, 2.0, // token 1, head 0
        1.0, 0.0, // token 1, head 1
    ];
    let key_cache = vec![
        1.0, 0.0, // slot 0
        0.0, 1.0, // slot 1
        1.0, 1.0, // slot 2
        -1.0, 0.5, // slot 3
    ];
    let weights = vec![
        1.0, 0.5, // token 0
        0.25, 2.0, // token 1
    ];
    let context_lens = vec![4, 2];
    (query, key_cache, weights, context_lens)
}

pub fn scores_close(actual: &[f32], expected: &[f32]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| (*actual - *expected).abs() < 1e-5)
}

fn v4_layout_matches(
    output: &[u8],
    block_size: usize,
    token_index: usize,
    nope: &[u8],
    rope: &[u16],
    scales: &[u8],
) -> bool {
    let expected_block_bytes = block_size * (V4_TOKEN_STRIDE + V4_SCALE_DIM);
    if output.len() != expected_block_bytes {
        return false;
    }
    let token_base = token_index * V4_TOKEN_STRIDE;
    let scale_base = block_size * V4_TOKEN_STRIDE + token_index * V4_SCALE_DIM;
    if &output[token_base..token_base + V4_NOPE_BYTES] != nope {
        return false;
    }
    for (idx, value) in rope.iter().copied().enumerate() {
        let base = token_base + V4_NOPE_BYTES + idx * 2;
        if output[base] != (value & 0xff) as u8 || output[base + 1] != (value >> 8) as u8 {
            return false;
        }
    }
    if &output[scale_base..scale_base + V4_SCALE_DIM] != scales {
        return false;
    }
    true
}

fn reference_compressed_state(
    input: &CudaDeepSeekCompressNormRopeFp8CacheInput<'_>,
    token_idx: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; input.head_size as usize];
    let position = input.positions[token_idx];
    let req_idx = input.token_to_req_indices[token_idx] as usize;
    let window_tokens = (1 + input.overlap) * input.compress_ratio;
    let start = position - window_tokens as i64 + 1;
    let row_stride = input.state_width as usize * 2;
    let block_stride = input.state_block_size as usize * row_stride;
    for dim in 0..input.head_size as usize {
        let mut max_score = f32::NEG_INFINITY;
        for window in 0..window_tokens as usize {
            let pos = start + window as i64;
            if pos < 0 {
                continue;
            }
            let block_idx = pos as usize / input.state_block_size as usize;
            if block_idx >= input.block_table_stride as usize {
                continue;
            }
            let block_number =
                input.block_table[req_idx * input.block_table_stride as usize + block_idx];
            if block_number < 0 {
                continue;
            }
            let block_offset = pos as usize % input.state_block_size as usize;
            let head_offset = if window as u32 >= input.compress_ratio {
                input.head_size as usize
            } else {
                0
            };
            let base =
                block_number as usize * block_stride + block_offset * row_stride + head_offset;
            max_score = max_score.max(input.state_cache[base + input.state_width as usize + dim]);
        }
        let mut weighted = 0.0f32;
        let mut denom = 0.0f32;
        for window in 0..window_tokens as usize {
            let pos = start + window as i64;
            if pos < 0 {
                continue;
            }
            let block_idx = pos as usize / input.state_block_size as usize;
            if block_idx >= input.block_table_stride as usize {
                continue;
            }
            let block_number =
                input.block_table[req_idx * input.block_table_stride as usize + block_idx];
            if block_number < 0 {
                continue;
            }
            let block_offset = pos as usize % input.state_block_size as usize;
            let head_offset = if window as u32 >= input.compress_ratio {
                input.head_size as usize
            } else {
                0
            };
            let base =
                block_number as usize * block_stride + block_offset * row_stride + head_offset;
            let score = input.state_cache[base + input.state_width as usize + dim];
            let weight = (score - max_score).exp();
            weighted += input.state_cache[base + dim] * weight;
            denom += weight;
        }
        out[dim] = if denom > 0.0 { weighted / denom } else { 0.0 };
    }
    out
}

fn reference_rope(
    input: &CudaDeepSeekCompressNormRopeFp8CacheInput<'_>,
    position: i64,
    normed: &[f32],
) -> Vec<f32> {
    let mut rotated = normed.to_vec();
    let nope = (input.head_size - input.rope_head_dim) as usize;
    let half_rope = input.rope_head_dim as usize / 2;
    let compressed_pos =
        (position as usize / input.compress_ratio as usize) * input.compress_ratio as usize;
    let cs_base = compressed_pos * input.cos_sin_stride as usize;
    for pair in 0..half_rope {
        let base = nope + pair * 2;
        let even = normed[base];
        let odd = normed[base + 1];
        let cos = input.cos_sin_cache[cs_base + pair];
        let sin = input.cos_sin_cache[cs_base + half_rope + pair];
        rotated[base] = even * cos - odd * sin;
        rotated[base + 1] = odd * cos + even * sin;
    }
    rotated
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn encode_e8m0_scale(scale: f32) -> u8 {
    (scale.log2().ceil() as i32 + 127).clamp(0, 255) as u8
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

fn f32_to_mxfp4_e2m1_nibble_nearest(value: f32) -> u8 {
    let mut best_bits = 0u8;
    let mut best_error = f32::INFINITY;
    for bits in 0u8..16 {
        let candidate = mxfp4_e2m1_nibble_to_f32(bits);
        let error = (candidate - value).abs();
        if error < best_error || (error == best_error && bits < best_bits) {
            best_error = error;
            best_bits = bits;
        }
    }
    best_bits
}

fn mxfp4_e2m1_nibble_to_f32(bits: u8) -> f32 {
    const TABLE: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    TABLE[(bits & 0x0f) as usize]
}
