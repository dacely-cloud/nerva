use nerva_cuda::experimental_rt::summary::CudaExperimentalRtCandidateBenchSummary;

pub(crate) fn page_level_equal_bytes_baseline_json(
    summary: &CudaExperimentalRtCandidateBenchSummary,
) -> String {
    let kv_bytes_per_token = saturating_mul(saturating_mul(u64::from(summary.dims), 2), 4);
    let page_kv_bytes = saturating_mul(u64::from(summary.page_tokens), kv_bytes_per_token);
    let local_kv_bytes = saturating_mul(summary.local_window_tokens, kv_bytes_per_token);
    let rt_far_kv_bytes = saturating_mul(
        saturating_mul(
            u64::from(summary.candidates_per_query),
            u64::from(summary.page_tokens),
        ),
        kv_bytes_per_token,
    );
    let candidate_pages = if page_kv_bytes == 0 {
        0
    } else {
        rt_far_kv_bytes / page_kv_bytes
    };
    let far_kv_bytes = saturating_mul(candidate_pages, page_kv_bytes);
    let total_kv_bytes = local_kv_bytes.saturating_add(far_kv_bytes);
    let dense_kv_bytes = saturating_mul(
        saturating_mul(u64::from(summary.pages), u64::from(summary.page_tokens)),
        kv_bytes_per_token,
    );

    format!(
        "{{\"scope\":\"synthetic_equal_bytes_measured\",\"selector\":\"synthetic_page_level_descriptor_topk\",\"quest_reproduction_status\":\"pending\",\"candidate_pages_per_query\":{},\"far_kv_bytes_per_query\":{},\"estimated_attention_kv_bytes_per_query\":{},\"estimated_vs_dense_kv_fraction_ppm\":{},\"attention_mass_recall_min_ppm\":{},\"attention_mass_recall_avg_ppm\":{},\"far_oracle_topk_token_recall_min_ppm\":{},\"far_oracle_topk_token_recall_avg_ppm\":{}}}",
        candidate_pages,
        far_kv_bytes,
        total_kv_bytes,
        fraction_ppm(total_kv_bytes, dense_kv_bytes),
        summary.page_level_attention_mass_recall_min_ppm,
        summary.page_level_attention_mass_recall_avg_ppm,
        summary.page_level_far_oracle_topk_token_recall_min_ppm,
        summary.page_level_far_oracle_topk_token_recall_avg_ppm,
    )
}

fn saturating_mul(lhs: u64, rhs: u64) -> u64 {
    let product = u128::from(lhs) * u128::from(rhs);
    product.min(u128::from(u64::MAX)) as u64
}

fn fraction_ppm(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return 0;
    }
    let scaled = u128::from(numerator) * 1_000_000u128;
    (scaled / u128::from(denominator)).min(u128::from(u64::MAX)) as u64
}
