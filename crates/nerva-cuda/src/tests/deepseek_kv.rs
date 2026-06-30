use crate::deepseek_kv::c128_topk::deepseek_c128_topk_metadata;
use crate::deepseek_kv::pack::deepseek_fp8_ds_mla_pack;
use crate::deepseek_kv::probe::{
    deepseek_c128_topk_metadata_smoke, deepseek_compressed_slot_mapping_smoke, deepseek_kv_smoke,
};
use crate::deepseek_kv::slot_mapping::deepseek_compressed_slot_mapping;
use crate::deepseek_kv::summary::{
    CudaDeepSeekC128TopkMetadataSummary, CudaDeepSeekCompressedSlotMappingSummary,
    CudaDeepSeekKvSummary,
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
