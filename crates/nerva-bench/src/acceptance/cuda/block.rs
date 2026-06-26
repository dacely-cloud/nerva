use nerva_model::precision::file_smoke::PrecisionSafetensorsBlockSmokeSummary;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_precision_checks(
    report: &mut AcceptanceReport,
    summary: &PrecisionSafetensorsBlockSmokeSummary,
) {
    let cuda_block = nerva_runtime::engine::cuda::cuda_tiny_block_smoke();
    let cuda_block_passed = format!("{:?}", cuda_block.status) == "Ok"
        && cuda_block.hidden == summary.hidden as u32
        && cuda_block.intermediate == summary.intermediate as u32
        && cuda_block.output_hash == summary.expected_hash
        && cuda_block.kernel_launches == 1
        && cuda_block.sync_calls == 1
        && cuda_block.d2h_bytes == 4
        && cuda_block.device_arena_bytes == 4
        && cuda_block.pinned_host_bytes == 4
        && cuda_block.hot_path_allocations == 0;
    report.push(
        "cuda_real_block",
        cuda_block_passed,
        format!(
            "status={:?} hidden={} intermediate={} output_hash={} expected_hash={} output_bits=[{},{}] kernel_launches={} sync_calls={} D2H_bytes={} device_arena_bytes={} pinned_host_bytes={} hot_path_allocations={} error={}",
            cuda_block.status,
            cuda_block.hidden,
            cuda_block.intermediate,
            cuda_block.output_hash,
            summary.expected_hash,
            cuda_block.output[0],
            cuda_block.output[1],
            cuda_block.kernel_launches,
            cuda_block.sync_calls,
            cuda_block.d2h_bytes,
            cuda_block.device_arena_bytes,
            cuda_block.pinned_host_bytes,
            cuda_block.hot_path_allocations,
            cuda_block.error.as_deref().unwrap_or("none"),
        ),
    );

    let cuda_resident_block = nerva_runtime::engine::cuda::cuda_loaded_tiny_block_smoke();
    let cuda_resident_block_passed = format!("{:?}", cuda_resident_block.status) == "Ok"
        && cuda_resident_block.hidden == summary.hidden as u32
        && cuda_resident_block.intermediate == summary.intermediate as u32
        && cuda_resident_block.output_hash == summary.expected_hash
        && cuda_resident_block.output == cuda_block.output
        && cuda_resident_block.resident_weight_bytes == summary.bytes_loaded as u64
        && cuda_resident_block.device_arena_bytes >= cuda_resident_block.resident_weight_bytes + 8
        && cuda_resident_block.pinned_host_bytes == cuda_resident_block.device_arena_bytes
        && cuda_resident_block.h2d_bytes == cuda_resident_block.device_arena_bytes
        && cuda_resident_block.d2h_bytes == 4
        && cuda_resident_block.kernel_launches == 1
        && cuda_resident_block.sync_calls == 2
        && cuda_resident_block.hot_path_allocations == 0;
    report.push(
        "cuda_resident_block",
        cuda_resident_block_passed,
        format!(
            "status={:?} hidden={} intermediate={} output_hash={} expected_hash={} output_bits=[{},{}] resident_weight_bytes={} H2D_bytes={} D2H_bytes={} device_arena_bytes={} pinned_host_bytes={} kernel_launches={} sync_calls={} hot_path_allocations={} error={}",
            cuda_resident_block.status,
            cuda_resident_block.hidden,
            cuda_resident_block.intermediate,
            cuda_resident_block.output_hash,
            summary.expected_hash,
            cuda_resident_block.output[0],
            cuda_resident_block.output[1],
            cuda_resident_block.resident_weight_bytes,
            cuda_resident_block.h2d_bytes,
            cuda_resident_block.d2h_bytes,
            cuda_resident_block.device_arena_bytes,
            cuda_resident_block.pinned_host_bytes,
            cuda_resident_block.kernel_launches,
            cuda_resident_block.sync_calls,
            cuda_resident_block.hot_path_allocations,
            cuda_resident_block.error.as_deref().unwrap_or("none"),
        ),
    );
}

pub(crate) fn push_prerequisite_failure(report: &mut AcceptanceReport, details: &str) {
    report.push(
        "cuda_real_block",
        false,
        format!("canonical precision block prerequisite failed: {details}"),
    );
    report.push(
        "cuda_resident_block",
        false,
        format!("canonical precision block prerequisite failed: {details}"),
    );
}
