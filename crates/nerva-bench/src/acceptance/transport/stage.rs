use nerva_runtime::engine::runtime::Runtime;
use nerva_runtime::transport::stage::config::StagePipelineConfig;
use nerva_runtime::transport::stage::summary::StagePipelineStatus;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_stage_pipeline(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_stage_pipeline_probe(StagePipelineConfig::reference_decode()) {
        Ok(summary) => report.push(
            "stage_pipeline_activation_only",
            matches!(summary.status, StagePipelineStatus::Ok)
                && summary.passed()
                && summary.boundaries > 0
                && summary.activation_only_boundaries == summary.boundaries
                && summary.inter_stage_weight_bytes == 0
                && summary.all_reduce_bytes == 0
                && summary.pageable_copies == 0
                && summary.per_token_registrations == 0
                && summary.hot_path_allocations == 0,
            format!(
                "stages={} boundaries={} activation_bytes_per_boundary={} total_activation_tx_bytes={} stage_local_weight_bytes={} stage_local_kv_bytes={} host_staged_boundaries={} gpu_direct_boundaries={} fallback_decisions={} inter_stage_weight_bytes={} all_reduce_bytes={} phase_handoff_syncs={} hot_path_allocations={}",
                summary.stages,
                summary.boundaries,
                summary.activation_bytes_per_boundary,
                summary.total_activation_tx_bytes,
                summary.stage_local_weight_bytes,
                summary.stage_local_kv_bytes,
                summary.host_staged_boundaries,
                summary.gpu_direct_boundaries,
                summary.fallback_decisions,
                summary.inter_stage_weight_bytes,
                summary.all_reduce_bytes,
                summary.phase_handoff_syncs,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("stage_pipeline_activation_only", false, format!("{err:?}")),
    }
}
