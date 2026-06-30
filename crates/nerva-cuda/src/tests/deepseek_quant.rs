use crate::deepseek_quant::dequant::{
    deepseek_fp8_e4m3fn_e8m0_dequant, deepseek_mxfp4_e2m1_e8m0_dequant,
};
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

#[test]
fn deepseek_quant_dequant_apis_match_reference_values() {
    let _guard = super::cuda_lock::cuda_test_lock();

    let fp8_weights = [
        0x38, 0x40, 0x30, 0xb8, 0x70, 0x77, 0x78, 0x7e, 0x20, 0x28, 0x30, 0x38,
    ];
    let fp8_scales = [0x7f, 0x80, 0x7e, 0x81];
    let fp8 = deepseek_fp8_e4m3fn_e8m0_dequant(&fp8_weights, &fp8_scales, 3, 4, 2, 2);
    if fp8.status != SmokeStatus::Ok {
        return;
    }
    assert_eq!(fp8.rows, 3);
    assert_eq!(fp8.cols, 4);
    assert_eq!(fp8.block_rows, 2);
    assert_eq!(fp8.block_cols, 2);
    assert_eq!(
        fp8.output,
        [
            1.0, 2.0, 1.0, -2.0, 128.0, 240.0, 512.0, 896.0, 0.0625, 0.125, 2.0, 4.0
        ]
    );
    assert_eq!(fp8.kernel_launches, 1);
    assert_eq!(fp8.sync_calls, 1);
    assert_eq!(fp8.hot_path_allocations, 0);
    assert!(fp8.output_hash != 0);

    let mxfp4_packed = [0x21, 0x76, 0xa9, 0xfe, 0x10, 0x54, 0x98, 0xdc];
    let mxfp4_scales = [0x7f, 0x80, 0x7e, 0x81];
    let mxfp4 = deepseek_mxfp4_e2m1_e8m0_dequant(&mxfp4_packed, &mxfp4_scales, 2, 4, 2);
    assert_eq!(mxfp4.status, SmokeStatus::Ok, "mxfp4 dequant: {mxfp4:?}");
    assert_eq!(mxfp4.rows, 2);
    assert_eq!(mxfp4.cols, 8);
    assert_eq!(mxfp4.block_rows, 1);
    assert_eq!(mxfp4.block_cols, 4);
    assert_eq!(
        mxfp4.output,
        [
            0.5, 1.0, 4.0, 6.0, -1.0, -2.0, -8.0, -12.0, 0.0, 0.25, 1.0, 1.5, -0.0, -2.0, -8.0,
            -12.0,
        ]
    );
    assert_eq!(mxfp4.kernel_launches, 1);
    assert_eq!(mxfp4.sync_calls, 1);
    assert_eq!(mxfp4.hot_path_allocations, 0);
    assert!(mxfp4.output_hash != 0);
}
