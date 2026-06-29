use crate::experimental_rt::summary::CudaExperimentalRtCandidateBenchSummary;
use crate::smoke::status::SmokeStatus;

pub(crate) fn summary_passed(summary: &CudaExperimentalRtCandidateBenchSummary) -> bool {
    summary.status == SmokeStatus::Ok
        && summary.pages > 0
        && summary.page_tokens > 0
        && summary.dims > 0
        && summary.query_count > 0
        && summary.candidates_per_query > 0
        && summary.iterations > 0
        && summary.dense_selector_avg_ns > 0
        && summary.candidate_selector_avg_ns > 0
        && summary.rerank_avg_ns > 0
        && summary.selector_plus_rerank_avg_ns > 0
        && summary.local_attention_avg_ns > 0
        && summary.kv_page_access_avg_ns > 0
        && summary.far_sparse_attention_avg_ns > 0
        && summary.softmax_merge_avg_ns > 0
        && summary.dense_full_attention_avg_ns > 0
        && summary.attention_mass_recall_min_ppm > 0
        && summary.attention_mass_recall_avg_ppm > 0
        && summary.page_level_attention_mass_recall_min_ppm > 0
        && summary.page_level_attention_mass_recall_avg_ppm > 0
        && summary.far_oracle_topk_tokens > 0
        && summary.far_oracle_topk_importance_scatter_min_pages > 0
        && summary.far_oracle_topk_importance_scatter_avg_pages_x1000 > 0
        && summary.far_oracle_topk_importance_scatter_max_pages > 0
        && summary.dense_selector_attention_stage_avg_ns > 0
        && summary.rt_selector_attention_stage_avg_ns > 0
        && summary.rt_selector_overlapped_attention_stage_avg_ns > 0
        && (!summary.real_rt_backend_available || summary.candidate_parity_checked)
        && summary.candidate_parity_mismatches == 0
        && summary.device_allocations == summary.device_frees
        && summary.hot_path_allocations == 0
}
