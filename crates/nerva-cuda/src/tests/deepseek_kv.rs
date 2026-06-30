use crate::deepseek_kv::c4_indexer_topk::deepseek_c4_indexer_topk;
use crate::deepseek_kv::c128_topk::deepseek_c128_topk_metadata;
use crate::deepseek_kv::compress_cache::{
    DEEPSEEK_COMPRESS_SCALE_E8M0, DEEPSEEK_COMPRESS_SCALE_F32, DEEPSEEK_COMPRESS_SCALE_MXFP4,
    deepseek_compress_norm_rope_fp8_cache,
};
use crate::deepseek_kv::pack::deepseek_fp8_ds_mla_pack;
use crate::deepseek_kv::partial_states::deepseek_save_partial_states;
use crate::deepseek_kv::probe::{
    c4_indexer_topk_fixture, compress_cache_fixture, deepseek_c4_indexer_topk_smoke,
    deepseek_c128_topk_metadata_smoke, deepseek_compress_norm_rope_fp8_cache_smoke,
    deepseek_compress_norm_rope_mxfp4_cache_smoke, deepseek_compressed_slot_mapping_smoke,
    deepseek_kv_smoke, deepseek_save_partial_states_smoke, mxfp4_compress_cache_fixture,
    reference_compress_norm_rope_fp8_cache, scores_close,
};
use crate::deepseek_kv::slot_mapping::deepseek_compressed_slot_mapping;
use crate::deepseek_kv::summary::{
    CudaDeepSeekC4IndexerTopkSummary, CudaDeepSeekC128TopkMetadataSummary,
    CudaDeepSeekCompressNormRopeFp8CacheSummary, CudaDeepSeekCompressedSlotMappingSummary,
    CudaDeepSeekKvSummary, CudaDeepSeekSavePartialStatesSummary,
};
use crate::smoke::status::SmokeStatus;

#[test]
fn deepseek_kv_summary_serializes_vllm_layout_metrics() {
    let summary = CudaDeepSeekKvSummary {
        status: SmokeStatus::Ok,
        return_code: 0,
        cuda_error: 0,
        block_size: 4,
        token_index: 2,
        token_stride: 576,
        scale_dim: 8,
        block_bytes: 2336,
        output_hash: 77,
        output: vec![0; 2336],
        device_arena_bytes: 3000,
        pinned_host_bytes: 2336,
        h2d_bytes: 584,
        d2h_bytes: 2336,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"token_stride\":576"));
    assert!(json.contains("\"scale_dim\":8"));
    assert!(json.contains("\"block_bytes\":2336"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn deepseek_compressed_slot_mapping_summary_serializes_vllm_metadata() {
    let summary = CudaDeepSeekCompressedSlotMappingSummary {
        status: SmokeStatus::Ok,
        return_code: 0,
        cuda_error: 0,
        num_tokens: 9,
        num_reqs: 2,
        block_table_stride: 4,
        block_size: 4,
        compress_ratio: 4,
        valid_slots: 2,
        pad_slots: 7,
        output_hash: 99,
        output_slots: vec![-1, -1, 81, -1, -1, 120, -1, -1, -1],
        device_arena_bytes: 200,
        pinned_host_bytes: 72,
        h2d_bytes: 60,
        d2h_bytes: 72,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"num_tokens\":9"));
    assert!(json.contains("\"compress_ratio\":4"));
    assert!(json.contains("\"valid_slots\":2"));
    assert!(json.contains("\"pad_slots\":7"));
}

#[test]
fn deepseek_c128_topk_metadata_summary_serializes_vllm_metadata() {
    let summary = CudaDeepSeekC128TopkMetadataSummary {
        status: SmokeStatus::Ok,
        return_code: 0,
        cuda_error: 0,
        num_tokens: 4,
        num_decode_tokens: 2,
        num_prefill_tokens: 2,
        num_reqs: 2,
        block_table_stride: 4,
        block_size: 2,
        compress_ratio: 128,
        max_compressed_tokens: 4,
        valid_decode_tokens: 1,
        decode_entries: 1,
        prefill_entries: 7,
        output_hash: 88,
        global_decode: vec![80, -1, -1, -1, 100, 101, -1, -1],
        decode_lens: vec![1, 0],
        prefill_local: vec![0, 1, 2, -1, 0, 1, 2, 3],
        device_arena_bytes: 300,
        pinned_host_bytes: 72,
        h2d_bytes: 128,
        d2h_bytes: 72,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"num_decode_tokens\":2"));
    assert!(json.contains("\"num_prefill_tokens\":2"));
    assert!(json.contains("\"compress_ratio\":128"));
    assert!(json.contains("\"prefill_entries\":7"));
}

#[test]
fn deepseek_c4_indexer_topk_summary_serializes_vllm_metadata() {
    let summary = CudaDeepSeekC4IndexerTopkSummary {
        status: SmokeStatus::Ok,
        return_code: 0,
        cuda_error: 0,
        num_tokens: 2,
        num_heads: 2,
        head_dim: 2,
        max_compressed_tokens: 4,
        topk_tokens: 2,
        valid_tokens: 2,
        selected_entries: 4,
        output_hash: 77,
        topk_indices: vec![2, 0, 0, 1],
        topk_scores: vec![1.5, 1.0, 2.0, 0.5],
        device_arena_bytes: 300,
        pinned_host_bytes: 32,
        h2d_bytes: 96,
        d2h_bytes: 32,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"num_heads\":2"));
    assert!(json.contains("\"head_dim\":2"));
    assert!(json.contains("\"topk_tokens\":2"));
    assert!(json.contains("\"selected_entries\":4"));
}

#[test]
fn deepseek_save_partial_states_summary_serializes_vllm_metadata() {
    let summary = CudaDeepSeekSavePartialStatesSummary {
        status: SmokeStatus::Ok,
        return_code: 0,
        cuda_error: 0,
        num_tokens: 3,
        block_size: 4,
        head_size: 3,
        state_width: 4,
        compress_ratio: 4,
        num_blocks: 2,
        written_tokens: 2,
        skipped_tokens: 1,
        output_hash: 123,
        state_cache: vec![0.0; 64],
        device_arena_bytes: 512,
        pinned_host_bytes: 256,
        h2d_bytes: 128,
        d2h_bytes: 256,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"head_size\":3"));
    assert!(json.contains("\"state_width\":4"));
    assert!(json.contains("\"written_tokens\":2"));
    assert!(json.contains("\"skipped_tokens\":1"));
}

#[test]
fn deepseek_compress_norm_rope_fp8_cache_summary_serializes_vllm_metadata() {
    let summary = CudaDeepSeekCompressNormRopeFp8CacheSummary {
        status: SmokeStatus::Ok,
        return_code: 0,
        cuda_error: 0,
        num_tokens: 2,
        head_size: 512,
        rope_head_dim: 64,
        compress_ratio: 128,
        quant_block: 64,
        token_stride: 576,
        scale_dim: 8,
        scale_format: DEEPSEEK_COMPRESS_SCALE_E8M0,
        written_tokens: 1,
        skipped_tokens: 1,
        kv_cache_bytes: 584,
        output_hash: 99,
        kv_cache: vec![0; 584],
        device_arena_bytes: 2048,
        pinned_host_bytes: 584,
        h2d_bytes: 1024,
        d2h_bytes: 584,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"token_stride\":576"));
    assert!(json.contains("\"scale_dim\":8"));
    assert!(json.contains("\"scale_format\":0"));
    assert!(json.contains("\"written_tokens\":1"));
}

#[test]
fn deepseek_compress_norm_rope_mxfp4_cache_summary_serializes_vllm_metadata() {
    let summary = CudaDeepSeekCompressNormRopeFp8CacheSummary {
        status: SmokeStatus::Ok,
        return_code: 0,
        cuda_error: 0,
        num_tokens: 2,
        head_size: 128,
        rope_head_dim: 64,
        compress_ratio: 128,
        quant_block: 32,
        token_stride: 64,
        scale_dim: 4,
        scale_format: DEEPSEEK_COMPRESS_SCALE_MXFP4,
        written_tokens: 1,
        skipped_tokens: 1,
        kv_cache_bytes: 272,
        output_hash: 99,
        kv_cache: vec![0; 272],
        device_arena_bytes: 2048,
        pinned_host_bytes: 272,
        h2d_bytes: 1024,
        d2h_bytes: 272,
        kernel_launches: 1,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"token_stride\":64"));
    assert!(json.contains("\"scale_dim\":4"));
    assert!(json.contains("\"scale_format\":2"));
}

#[test]
fn deepseek_kv_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_kv_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_kv_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.block_size, 4);
    assert_eq!(second.token_index, 2);
    assert_eq!(second.token_stride, 576);
    assert_eq!(second.scale_dim, 8);
    assert_eq!(second.block_bytes, 2336);
    assert_eq!(second.output_hash, first.output_hash);
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
}

#[test]
fn deepseek_compressed_slot_mapping_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_compressed_slot_mapping_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_compressed_slot_mapping_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.num_tokens, 9);
    assert_eq!(second.num_reqs, 2);
    assert_eq!(second.block_size, 4);
    assert_eq!(second.compress_ratio, 4);
    assert_eq!(second.valid_slots, 2);
    assert_eq!(second.pad_slots, 7);
    assert_eq!(
        second.output_slots,
        vec![-1, -1, 81, -1, -1, 120, -1, -1, -1]
    );
    assert_eq!(second.output_hash, first.output_hash);
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
}

#[test]
fn deepseek_c128_topk_metadata_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_c128_topk_metadata_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_c128_topk_metadata_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.num_tokens, 4);
    assert_eq!(second.num_decode_tokens, 2);
    assert_eq!(second.num_prefill_tokens, 2);
    assert_eq!(second.compress_ratio, 128);
    assert_eq!(second.global_decode, vec![80, -1, -1, -1, 100, 101, -1, -1]);
    assert_eq!(second.decode_lens, vec![1, 0]);
    assert_eq!(second.prefill_local, vec![0, 1, 2, -1, 0, 1, 2, 3]);
    assert_eq!(second.valid_decode_tokens, 1);
    assert_eq!(second.decode_entries, 1);
    assert_eq!(second.prefill_entries, 7);
    assert_eq!(second.output_hash, first.output_hash);
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
}

#[test]
fn deepseek_c4_indexer_topk_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_c4_indexer_topk_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_c4_indexer_topk_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.num_tokens, 2);
    assert_eq!(second.num_heads, 2);
    assert_eq!(second.head_dim, 2);
    assert_eq!(second.max_compressed_tokens, 4);
    assert_eq!(second.topk_tokens, 2);
    assert_eq!(second.topk_indices, vec![2, 0, 0, 1]);
    assert!(scores_close(&second.topk_scores, &[1.5, 1.0, 2.0, 0.5]));
    assert_eq!(second.valid_tokens, 2);
    assert_eq!(second.selected_entries, 4);
    assert_eq!(second.output_hash, first.output_hash);
    assert_eq!(second.kernel_launches, 2);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
}

#[test]
fn deepseek_save_partial_states_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_save_partial_states_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_save_partial_states_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.num_tokens, 3);
    assert_eq!(second.block_size, 4);
    assert_eq!(second.head_size, 3);
    assert_eq!(second.state_width, 4);
    assert_eq!(second.compress_ratio, 4);
    assert_eq!(second.written_tokens, 2);
    assert_eq!(second.skipped_tokens, 1);
    assert_partial_state_fixture(&second.state_cache);
    assert_eq!(second.output_hash, first.output_hash);
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
}

#[test]
fn deepseek_compress_norm_rope_fp8_cache_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_compress_norm_rope_fp8_cache_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_compress_norm_rope_fp8_cache_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.num_tokens, 2);
    assert_eq!(second.head_size, 4);
    assert_eq!(second.rope_head_dim, 2);
    assert_eq!(second.compress_ratio, 2);
    assert_eq!(second.scale_format, DEEPSEEK_COMPRESS_SCALE_E8M0);
    assert_eq!(second.written_tokens, 2);
    assert_eq!(second.skipped_tokens, 0);
    assert_eq!(second.output_hash, first.output_hash);
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
}

#[test]
fn deepseek_compress_norm_rope_mxfp4_cache_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_compress_norm_rope_mxfp4_cache_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_compress_norm_rope_mxfp4_cache_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.num_tokens, 2);
    assert_eq!(second.head_size, 128);
    assert_eq!(second.rope_head_dim, 64);
    assert_eq!(second.compress_ratio, 2);
    assert_eq!(second.quant_block, 32);
    assert_eq!(second.token_stride, 64);
    assert_eq!(second.scale_dim, 4);
    assert_eq!(second.scale_format, DEEPSEEK_COMPRESS_SCALE_MXFP4);
    assert_eq!(second.written_tokens, 2);
    assert_eq!(second.skipped_tokens, 0);
    assert_eq!(second.output_hash, first.output_hash);
    assert_eq!(second.kernel_launches, 1);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
}

#[test]
fn deepseek_fp8_ds_mla_pack_matches_vllm_block_offsets() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let block_size = 3;
    let token_index = 1;
    let nope = [0x10, 0x20, 0x30, 0x40];
    let rope = [0x3f80, 0x4000];
    let scales = [0x7f, 0x00];
    let summary = deepseek_fp8_ds_mla_pack(block_size, token_index, &nope, &rope, &scales);
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let token_stride = nope.len() + rope.len() * 2;
    let block_bytes = block_size as usize * (token_stride + scales.len());
    assert_eq!(summary.token_stride as usize, token_stride);
    assert_eq!(summary.block_bytes as usize, block_bytes);
    assert_eq!(summary.output.len(), block_bytes);

    let token_base = token_index as usize * token_stride;
    assert_eq!(&summary.output[token_base..token_base + nope.len()], &nope);
    assert_eq!(
        &summary.output[token_base + nope.len()..token_base + token_stride],
        &[0x80, 0x3f, 0x00, 0x40]
    );
    let scale_base = block_size as usize * token_stride + token_index as usize * scales.len();
    assert_eq!(
        &summary.output[scale_base..scale_base + scales.len()],
        &scales
    );
    assert!(summary.output[..token_base].iter().all(|byte| *byte == 0));
    assert!(summary.output_hash != 0);
}

#[test]
fn deepseek_save_partial_states_matches_vllm_state_cache_offsets() {
    let _guard = super::cuda_lock::cuda_test_lock();

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
        return;
    }

    assert_eq!(summary.written_tokens, 2);
    assert_eq!(summary.skipped_tokens, 1);
    assert_partial_state_fixture(&summary.state_cache);
    assert!(summary.output_hash != 0);
}

fn assert_partial_state_fixture(state_cache: &[f32]) {
    assert_eq!(state_cache.len(), 64);
    let row_stride = 8usize;
    let block_stride = 32usize;
    let token0_base = row_stride;
    assert_eq!(&state_cache[token0_base..token0_base + 3], &[1.0, 2.0, 3.0]);
    assert_close(
        &state_cache[token0_base + 4..token0_base + 7],
        &[40.1, 50.2, 60.3],
    );
    let token2_base = block_stride;
    assert_eq!(&state_cache[token2_base..token2_base + 3], &[7.0, 8.0, 9.0]);
    assert_close(
        &state_cache[token2_base + 4..token2_base + 7],
        &[100.7, 110.8, 120.9],
    );
    for (idx, value) in state_cache.iter().enumerate() {
        let in_token0 = (token0_base..token0_base + 3).contains(&idx)
            || (token0_base + 4..token0_base + 7).contains(&idx);
        let in_token2 = (token2_base..token2_base + 3).contains(&idx)
            || (token2_base + 4..token2_base + 7).contains(&idx);
        if !in_token0 && !in_token2 {
            assert_eq!(*value, 0.0, "unexpected state cache write at {idx}");
        }
    }
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!(
            (*actual - *expected).abs() < 1e-4,
            "actual {actual} expected {expected}"
        );
    }
}

#[test]
fn deepseek_c128_topk_metadata_matches_vllm_decode_and_prefill_math() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let positions = [127, 255, 383, 511];
    let token_to_req = [0, 1, 0, 1];
    let block_table = [
        40, 41, 42, 43, // request 0
        50, 51, 52, 53, // request 1
    ];
    let slot_mapping = [10, -1, 12, 13];
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
        return;
    }

    assert_eq!(
        summary.global_decode,
        vec![80, -1, -1, -1, 100, 101, -1, -1]
    );
    assert_eq!(summary.decode_lens, vec![1, 0]);
    assert_eq!(summary.prefill_local, vec![0, 1, 2, -1, 0, 1, 2, 3]);
    assert_eq!(summary.valid_decode_tokens, 1);
    assert_eq!(summary.decode_entries, 1);
    assert_eq!(summary.prefill_entries, 7);
    assert!(summary.output_hash != 0);
}

#[test]
fn deepseek_c4_indexer_topk_matches_vllm_local_weighted_score_math() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let (query, key_cache, weights, context_lens) = c4_indexer_topk_fixture();
    let summary = deepseek_c4_indexer_topk(&query, &key_cache, &weights, &context_lens, 2, 2, 2, 2);
    if summary.status != SmokeStatus::Ok {
        return;
    }

    assert_eq!(summary.topk_indices, vec![2, 0, 0, 1]);
    assert!(scores_close(&summary.topk_scores, &[1.5, 1.0, 2.0, 0.5]));
    assert_eq!(summary.valid_tokens, 2);
    assert_eq!(summary.selected_entries, 4);
    assert!(summary.output_hash != 0);
}

#[test]
fn deepseek_compressed_slot_mapping_matches_vllm_kernel_math() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let query_start_loc = [0, 5, 9];
    let seq_lens = [10, 7];
    let block_table = [
        20, 21, 22, 23, // request 0
        30, 31, 32, 33, // request 1
    ];
    let summary =
        deepseek_compressed_slot_mapping(&query_start_loc, &seq_lens, &block_table, 4, 4, 4);
    if summary.status != SmokeStatus::Ok {
        return;
    }

    assert_eq!(
        summary.output_slots,
        vec![-1, -1, 81, -1, -1, 120, -1, -1, -1]
    );
    assert_eq!(summary.valid_slots, 2);
    assert_eq!(summary.pad_slots, 7);
    assert!(summary.output_hash != 0);
}

#[test]
fn deepseek_compress_norm_rope_fp8_cache_matches_vllm_sparse_cache_math() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let fixture = compress_cache_fixture(DEEPSEEK_COMPRESS_SCALE_E8M0);
    let summary = deepseek_compress_norm_rope_fp8_cache(fixture.input.clone());
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected = reference_compress_norm_rope_fp8_cache(&fixture);
    assert_eq!(summary.kv_cache, expected);
    assert_eq!(summary.token_stride, 6);
    assert_eq!(summary.scale_dim, 2);
    assert_eq!(summary.scale_format, DEEPSEEK_COMPRESS_SCALE_E8M0);
    assert_eq!(summary.written_tokens, 2);
    assert_eq!(summary.skipped_tokens, 0);
    assert!(summary.output_hash != 0);
}

#[test]
fn deepseek_compress_norm_rope_fp8_cache_matches_vllm_indexer_cache_math() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let fixture = compress_cache_fixture(DEEPSEEK_COMPRESS_SCALE_F32);
    let summary = deepseek_compress_norm_rope_fp8_cache(fixture.input.clone());
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected = reference_compress_norm_rope_fp8_cache(&fixture);
    assert_eq!(summary.kv_cache, expected);
    assert_eq!(summary.token_stride, 4);
    assert_eq!(summary.scale_dim, size_of::<f32>() as u32);
    assert_eq!(summary.scale_format, DEEPSEEK_COMPRESS_SCALE_F32);
    assert_eq!(summary.written_tokens, 2);
    assert_eq!(summary.skipped_tokens, 0);
    assert!(summary.output_hash != 0);
}

#[test]
fn deepseek_compress_norm_rope_mxfp4_cache_matches_vllm_indexer_cache_math() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let fixture = mxfp4_compress_cache_fixture();
    let summary = deepseek_compress_norm_rope_fp8_cache(fixture.input.clone());
    if summary.status != SmokeStatus::Ok {
        return;
    }

    let expected = reference_compress_norm_rope_fp8_cache(&fixture);
    assert_eq!(summary.kv_cache, expected);
    assert_eq!(summary.head_size, 128);
    assert_eq!(summary.rope_head_dim, 64);
    assert_eq!(summary.quant_block, 32);
    assert_eq!(summary.token_stride, 64);
    assert_eq!(summary.scale_dim, 4);
    assert_eq!(summary.scale_format, DEEPSEEK_COMPRESS_SCALE_MXFP4);
    assert_eq!(summary.written_tokens, 2);
    assert_eq!(summary.skipped_tokens, 0);
    assert!(summary.output_hash != 0);
}
