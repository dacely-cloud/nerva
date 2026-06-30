use crate::deepseek_kv::c128_topk::deepseek_c128_topk_metadata;
use crate::deepseek_kv::pack::deepseek_fp8_ds_mla_pack;
use crate::deepseek_kv::partial_states::deepseek_save_partial_states;
use crate::deepseek_kv::slot_mapping::deepseek_compressed_slot_mapping;
use crate::deepseek_kv::summary::{
    CudaDeepSeekC128TopkMetadataSummary, CudaDeepSeekCompressedSlotMappingSummary,
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
    let expected = [-1, -1, 81, -1, -1, 120, -1, -1, -1];

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
