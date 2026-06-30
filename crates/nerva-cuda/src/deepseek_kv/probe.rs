use crate::deepseek_kv::c128_topk::{
    deepseek_c128_topk_metadata, deepseek_c128_topk_metadata_reference,
};
use crate::deepseek_kv::c4_indexer_topk::{
    deepseek_c4_indexer_topk, deepseek_c4_indexer_topk_reference,
};
use crate::deepseek_kv::compress_cache::{
    deepseek_compress_norm_rope_fp8_cache, deepseek_compress_norm_rope_fp8_cache_reference,
    CudaDeepSeekCompressNormRopeFp8CacheInput, DEEPSEEK_COMPRESS_SCALE_E8M0,
    DEEPSEEK_COMPRESS_SCALE_MXFP4,
};
use crate::deepseek_kv::pack::deepseek_fp8_ds_mla_pack;
use crate::deepseek_kv::partial_states::{
    deepseek_save_partial_states, deepseek_save_partial_states_reference,
};
use crate::deepseek_kv::slot_mapping::{
    deepseek_compressed_slot_mapping, deepseek_compressed_slot_mapping_reference,
};
use crate::deepseek_kv::summary::{
    CudaDeepSeekC128TopkMetadataSummary, CudaDeepSeekC4IndexerTopkSummary,
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
    let expected = match deepseek_c128_topk_metadata_reference(
        &positions,
        2,
        &token_to_req,
        &block_table,
        4,
        &slot_mapping,
        2,
        128,
        4,
    ) {
        Ok(expected) => expected,
        Err(err) => {
            let mut failed = deepseek_c128_topk_metadata(
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
            failed.status = SmokeStatus::Failed;
            failed.error = Some(format!(
                "DeepSeek C128A top-k metadata reference failed: {err}"
            ));
            return failed;
        }
    };

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

    if summary.global_decode != expected.global_decode
        || summary.decode_lens != expected.decode_lens
        || summary.prefill_local != expected.prefill_local
        || summary.valid_decode_tokens != expected.valid_decode_tokens
        || summary.decode_entries != expected.decode_entries
        || summary.prefill_entries != expected.prefill_entries
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
    let (expected_indices, expected_scores) = match deepseek_c4_indexer_topk_reference(
        &query,
        &key_cache,
        &weights,
        &context_lens,
        2,
        2,
        2,
        2,
    ) {
        Ok(expected) => expected,
        Err(err) => {
            let mut failed =
                deepseek_c4_indexer_topk(&query, &key_cache, &weights, &context_lens, 2, 2, 2, 2);
            failed.status = SmokeStatus::Failed;
            failed.error = Some(format!("DeepSeek C4 indexer top-k reference failed: {err}"));
            return failed;
        }
    };
    let summary = deepseek_c4_indexer_topk(&query, &key_cache, &weights, &context_lens, 2, 2, 2, 2);
    if summary.status != SmokeStatus::Ok {
        return summary;
    }

    if summary.topk_indices != expected_indices
        || !scores_close(&summary.topk_scores, &expected_scores)
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
    let expected = match deepseek_save_partial_states_reference(
        &kv,
        &score,
        &ape,
        &positions,
        &slot_mapping,
        4,
        3,
        4,
        4,
        2,
    ) {
        Ok(expected) => expected,
        Err(err) => {
            let mut failed = deepseek_save_partial_states(
                &kv,
                &score,
                &ape,
                &positions,
                &slot_mapping,
                4,
                3,
                4,
                4,
                2,
            );
            failed.status = SmokeStatus::Failed;
            failed.error = Some(format!(
                "DeepSeek save partial states reference failed: {err}"
            ));
            return failed;
        }
    };
    let summary =
        deepseek_save_partial_states(&kv, &score, &ape, &positions, &slot_mapping, 4, 3, 4, 4, 2);
    if summary.status != SmokeStatus::Ok {
        return summary;
    }

    if !f32_slices_close(&summary.state_cache, &expected.state_cache)
        || summary.written_tokens != expected.written_tokens
        || summary.skipped_tokens != expected.skipped_tokens
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

    let expected = match deepseek_compress_norm_rope_fp8_cache_reference(&fixture.input) {
        Ok(expected) => expected,
        Err(err) => {
            let mut failed = summary;
            failed.status = SmokeStatus::Failed;
            failed.error = Some(format!(
                "DeepSeek fused compress/norm/RoPE FP8 cache reference failed: {err}"
            ));
            return failed;
        }
    };
    if summary.kv_cache == expected.kv_cache
        && summary.written_tokens == expected.written_tokens
        && summary.skipped_tokens == expected.skipped_tokens
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

    let expected = match deepseek_compress_norm_rope_fp8_cache_reference(&fixture.input) {
        Ok(expected) => expected,
        Err(err) => {
            let mut failed = summary;
            failed.status = SmokeStatus::Failed;
            failed.error = Some(format!(
                "DeepSeek fused compress/norm/RoPE MXFP4 cache reference failed: {err}"
            ));
            return failed;
        }
    };
    if summary.kv_cache == expected.kv_cache
        && summary.written_tokens == expected.written_tokens
        && summary.skipped_tokens == expected.skipped_tokens
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
    f32_slices_close(actual, expected)
}

fn f32_slices_close(actual: &[f32], expected: &[f32]) -> bool {
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
