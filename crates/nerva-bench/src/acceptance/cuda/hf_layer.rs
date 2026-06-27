use nerva_core::types::id::token::TokenId;
use nerva_model::causal_lm::smoke::load_hf_causal_lm_smoke_fixture;
use nerva_runtime::engine::hf_cuda::run_loaded_hf_layer_on_cuda;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_loaded_hf_layer_forward(report: &mut AcceptanceReport) {
    let loaded = match load_hf_causal_lm_smoke_fixture() {
        Ok(loaded) => loaded,
        Err(err) => {
            report.push(
                "cuda_loaded_hf_layer_forward",
                false,
                format!("failed to load HF smoke fixture: {err:?}"),
            );
            return;
        }
    };
    let summary = match run_loaded_hf_layer_on_cuda(&loaded.model, 0, TokenId(0)) {
        Ok(summary) => summary,
        Err(err) => {
            report.push(
                "cuda_loaded_hf_layer_forward",
                false,
                format!("failed to execute HF layer on CUDA: {err:?}"),
            );
            return;
        }
    };

    report.push(
        "cuda_loaded_hf_layer_forward",
        summary.passed(),
        format!(
            "status={:?} layer={} token={} hidden={} output_hash={} expected_hash={} bit_parity={} resident_weight_bytes={} H2D_bytes={} D2H_bytes={} device_arena_bytes={} pinned_host_bytes={} kernel_launches={} sync_calls={} hot_path_allocations={} error={}",
            summary.cuda.status,
            summary.layer_index,
            summary.token.0,
            summary.hidden,
            summary.output_hash,
            summary.expected_hash,
            summary.bit_parity,
            summary.cuda.resident_weight_bytes,
            summary.cuda.h2d_bytes,
            summary.cuda.d2h_bytes,
            summary.cuda.device_arena_bytes,
            summary.cuda.pinned_host_bytes,
            summary.cuda.kernel_launches,
            summary.cuda.sync_calls,
            summary.hot_path_allocations,
            summary.cuda.error.as_deref().unwrap_or("none"),
        ),
    );
}
