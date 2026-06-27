use nerva_core::types::dtype::DType;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_model::common::shape::TransformerBlockShape;
use nerva_model::precision::bits::f32_to_f16_bits;
use nerva_model::precision::block::model::PrecisionTransformerBlock;
use nerva_model::precision::file_smoke::summary::PrecisionSafetensorsBlockSmokeSummary;
use nerva_model::precision::scratch::PrecisionTransformerBlockScratch;
use nerva_runtime::engine::cuda_block::run_precision_block_on_cuda;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_precision_checks(
    report: &mut AcceptanceReport,
    summary: &PrecisionSafetensorsBlockSmokeSummary,
) {
    let cuda_block = nerva_cuda::block::probe::tiny_block_smoke();
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

    let cuda_resident_block = nerva_cuda::block::probe::loaded_tiny_block_smoke();
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

    let generic = generic_precision_block_forward();
    report.push("cuda_loaded_precision_block_forward", generic.0, generic.1);
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
    report.push(
        "cuda_loaded_precision_block_forward",
        false,
        format!("canonical precision block prerequisite failed: {details}"),
    );
}

fn generic_precision_block_forward() -> (bool, String) {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let block = tiny_block(shape);
    let input = [f32_to_f16_bits(1.0), f32_to_f16_bits(2.0)];
    let mut scratch = PrecisionTransformerBlockScratch::new(shape).unwrap();
    let mut expected = [0u16; 2];
    let mut ledger = TokenLedger::new(0);
    if let Err(err) = block.forward_into(&input, &mut scratch, &mut expected, &mut ledger) {
        return (false, format!("CPU precision block failed: {err:?}"));
    }
    let summary = match run_precision_block_on_cuda(&block, &input, 0) {
        Ok(summary) => summary,
        Err(err) => {
            return (
                false,
                format!("CUDA precision block planning failed: {err:?}"),
            );
        }
    };
    let passed = format!("{:?}", summary.status) == "Ok"
        && summary.output == expected
        && summary.kernel_launches == 1
        && summary.sync_calls == 1
        && summary.hot_path_allocations == 0
        && summary.resident_weight_bytes > 0
        && summary.h2d_bytes >= summary.resident_weight_bytes
        && summary.d2h_bytes == 4;
    (
        passed,
        format!(
            "status={:?} hidden={} heads={} kv_heads={} head_dim={} output_hash={} expected_bits=[{},{}] output_bits=[{},{}] resident_weight_bytes={} H2D_bytes={} D2H_bytes={} device_arena_bytes={} pinned_host_bytes={} kernel_launches={} sync_calls={} hot_path_allocations={} error={}",
            summary.status,
            summary.hidden,
            summary.heads,
            summary.kv_heads,
            summary.head_dim,
            summary.output_hash,
            expected[0],
            expected[1],
            summary.output.first().copied().unwrap_or_default(),
            summary.output.get(1).copied().unwrap_or_default(),
            summary.resident_weight_bytes,
            summary.h2d_bytes,
            summary.d2h_bytes,
            summary.device_arena_bytes,
            summary.pinned_host_bytes,
            summary.kernel_launches,
            summary.sync_calls,
            summary.hot_path_allocations,
            summary.error.as_deref().unwrap_or("none"),
        ),
    )
}

fn tiny_block(shape: TransformerBlockShape) -> PrecisionTransformerBlock {
    let rms = [1.0, 1.0];
    let identity = [1.0, 0.0, 0.0, 1.0];
    let gate = [0.5, 0.0, 0.0, 0.5];
    PrecisionTransformerBlock::new_from_f32(
        DType::F16,
        shape,
        &rms,
        &rms,
        &identity,
        &identity,
        &identity,
        &identity,
        &gate,
        &identity,
        &identity,
        1e-5,
    )
    .unwrap()
}
