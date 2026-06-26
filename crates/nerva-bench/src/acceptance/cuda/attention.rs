use nerva_model::attention::smoke::BlockwiseAttentionSmokeSummary;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_tiered_check(
    report: &mut AcceptanceReport,
    summary: &BlockwiseAttentionSmokeSummary,
) {
    let cuda_attention = nerva_runtime::engine::cuda::cuda_tiered_attention_smoke();
    let max_abs_error = cuda_attention
        .output
        .iter()
        .zip(summary.output.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let cuda_attention_passed = format!("{:?}", cuda_attention.status) == "Ok"
        && cuda_attention.hidden == summary.hidden as u32
        && cuda_attention.heads == summary.heads as u32
        && cuda_attention.blocks == summary.blocks as u32
        && cuda_attention.tokens == summary.tokens as u32
        && max_abs_error <= 1.0e-6
        && cuda_attention.cpu_block_events == 1
        && cuda_attention.device_block_events == 1
        && cuda_attention.resident_kv_bytes == 32
        && cuda_attention.h2d_bytes >= cuda_attention.resident_kv_bytes
        && cuda_attention.d2h_bytes > 0
        && cuda_attention.kernel_launches == 1
        && cuda_attention.sync_calls == 1
        && cuda_attention.hot_path_allocations == 0;
    report.push(
        "cuda_tiered_attention",
        cuda_attention_passed,
        format!(
            "status={:?} hidden={} heads={} blocks={} tokens={} output=[{},{}] reference=[{},{}] max_abs_error={} output_hash={} reference_hash={} cpu_block_events={} device_block_events={} resident_kv_bytes={} H2D_bytes={} D2H_bytes={} kernel_launches={} sync_calls={} hot_path_allocations={} error={}",
            cuda_attention.status,
            cuda_attention.hidden,
            cuda_attention.heads,
            cuda_attention.blocks,
            cuda_attention.tokens,
            cuda_attention.output[0],
            cuda_attention.output[1],
            summary.output[0],
            summary.output[1],
            max_abs_error,
            cuda_attention.output_hash,
            summary.output_hash,
            cuda_attention.cpu_block_events,
            cuda_attention.device_block_events,
            cuda_attention.resident_kv_bytes,
            cuda_attention.h2d_bytes,
            cuda_attention.d2h_bytes,
            cuda_attention.kernel_launches,
            cuda_attention.sync_calls,
            cuda_attention.hot_path_allocations,
            cuda_attention.error.as_deref().unwrap_or("none"),
        ),
    );
}

pub(crate) fn push_prerequisite_failure(report: &mut AcceptanceReport, details: &str) {
    report.push(
        "cuda_tiered_attention",
        false,
        format!("tiered reference attention prerequisite failed: {details}"),
    );
}
