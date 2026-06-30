use crate::deepseek_kv::pack::deepseek_fp8_ds_mla_pack;
use crate::deepseek_kv::probe::deepseek_kv_smoke;
use crate::deepseek_kv::summary::CudaDeepSeekKvSummary;
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
