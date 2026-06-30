use crate::deepseek_quant::probe::deepseek_quant_smoke;
use crate::deepseek_quant::summary::CudaDeepSeekQuantSummary;
use crate::smoke::status::SmokeStatus;

#[test]
fn deepseek_quant_summary_serializes_layout_and_mismatches() {
    let summary = CudaDeepSeekQuantSummary {
        status: SmokeStatus::Ok,
        fp8_rows: 3,
        fp8_cols: 4,
        fp8_block_rows: 2,
        fp8_block_cols: 2,
        mxfp4_rows: 2,
        mxfp4_packed_cols: 4,
        mxfp4_scale_packed_cols: 2,
        fp8_output_hash: 11,
        mxfp4_output_hash: 22,
        fp8_mismatches: 0,
        mxfp4_mismatches: 0,
        fp8_max_abs_diff: 0.0,
        mxfp4_max_abs_diff: 0.0,
        device_arena_bytes: 128,
        pinned_host_bytes: 112,
        h2d_bytes: 28,
        d2h_bytes: 112,
        kernel_launches: 2,
        sync_calls: 1,
        hot_path_allocations: 0,
        error: None,
    };

    let json = summary.to_json();
    assert!(json.contains("\"status\":\"ok\""));
    assert!(json.contains("\"fp8_rows\":3"));
    assert!(json.contains("\"fp8_block_cols\":2"));
    assert!(json.contains("\"mxfp4_packed_cols\":4"));
    assert!(json.contains("\"fp8_mismatches\":0"));
    assert!(json.contains("\"mxfp4_mismatches\":0"));
    assert!(json.contains("\"kernel_launches\":2"));
    assert!(json.contains("\"hot_path_allocations\":0"));
}

#[test]
fn deepseek_quant_smoke_is_repeatable_when_device_is_available() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let first = deepseek_quant_smoke();
    if first.status != SmokeStatus::Ok {
        return;
    }

    let second = deepseek_quant_smoke();
    assert_eq!(second.status, SmokeStatus::Ok, "second smoke: {second:?}");
    assert_eq!(second.fp8_mismatches, 0);
    assert_eq!(second.mxfp4_mismatches, 0);
    assert_eq!(second.fp8_max_abs_diff, 0.0);
    assert_eq!(second.mxfp4_max_abs_diff, 0.0);
    assert_eq!(second.kernel_launches, 2);
    assert_eq!(second.sync_calls, 1);
    assert_eq!(second.hot_path_allocations, 0);
    assert_eq!(second.fp8_output_hash, first.fp8_output_hash);
    assert_eq!(second.mxfp4_output_hash, first.mxfp4_output_hash);
}
