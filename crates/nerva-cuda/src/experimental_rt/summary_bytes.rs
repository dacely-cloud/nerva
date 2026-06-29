use crate::experimental_rt::summary::CudaExperimentalRtCandidateBenchSummary;

pub(crate) struct SyntheticKvByteEstimates {
    pub kv_bytes_per_token: u64,
    pub dense_full_kv_bytes_per_query: u64,
    pub local_window_kv_bytes_per_query: u64,
    pub candidate_page_kv_bytes_per_query: u64,
    pub estimated_rt_attention_kv_bytes_per_query: u64,
    pub estimated_rt_vs_dense_kv_fraction_ppm: u64,
    pub far_oracle_topk_token_kv_bytes_per_query: u64,
    pub far_oracle_topk_scatter_page_kv_bytes_min_per_query: u64,
    pub far_oracle_topk_scatter_page_kv_bytes_avg_per_query: u64,
    pub far_oracle_topk_scatter_page_kv_bytes_max_per_query: u64,
    pub far_oracle_topk_page_overfetch_avg_x1000: u64,
    pub fine_token_projected_candidate_kv_bytes_per_query: u64,
    pub fine_token_projected_vs_page_candidate_kv_fraction_ppm: u64,
    pub fine_token_learned_projected_candidate_kv_bytes_per_query: u64,
    pub fine_token_learned_projected_vs_page_candidate_kv_fraction_ppm: u64,
}

pub(crate) fn synthetic_kv_byte_estimates(
    summary: &CudaExperimentalRtCandidateBenchSummary,
) -> SyntheticKvByteEstimates {
    let kv_bytes_per_token =
        saturating_mul(saturating_mul(u64::from(summary.dims), 2), f32_bytes());
    let dense_full_kv_bytes_per_query = saturating_mul(
        saturating_mul(u64::from(summary.pages), u64::from(summary.page_tokens)),
        kv_bytes_per_token,
    );
    let local_window_kv_bytes_per_query =
        saturating_mul(summary.local_window_tokens, kv_bytes_per_token);
    let candidate_page_kv_bytes_per_query = saturating_mul(
        saturating_mul(
            u64::from(summary.candidates_per_query),
            u64::from(summary.page_tokens),
        ),
        kv_bytes_per_token,
    );
    let estimated_rt_attention_kv_bytes_per_query =
        local_window_kv_bytes_per_query.saturating_add(candidate_page_kv_bytes_per_query);
    let page_kv_bytes_per_query =
        saturating_mul(u64::from(summary.page_tokens), kv_bytes_per_token);
    let far_oracle_topk_token_kv_bytes_per_query =
        saturating_mul(summary.far_oracle_topk_tokens, kv_bytes_per_token);
    let far_oracle_topk_scatter_page_kv_bytes_min_per_query = saturating_mul(
        summary.far_oracle_topk_importance_scatter_min_pages,
        page_kv_bytes_per_query,
    );
    let far_oracle_topk_scatter_page_kv_bytes_avg_per_query = saturating_mul_div(
        summary.far_oracle_topk_importance_scatter_avg_pages_x1000,
        page_kv_bytes_per_query,
        1_000,
    );
    let far_oracle_topk_scatter_page_kv_bytes_max_per_query = saturating_mul(
        summary.far_oracle_topk_importance_scatter_max_pages,
        page_kv_bytes_per_query,
    );
    let fine_token_projected_candidate_kv_bytes_per_query = saturating_mul(
        summary.fine_token_projected_candidate_tokens,
        kv_bytes_per_token,
    );
    let fine_token_learned_projected_candidate_kv_bytes_per_query = saturating_mul(
        summary.fine_token_learned_projected_candidate_tokens,
        kv_bytes_per_token,
    );

    SyntheticKvByteEstimates {
        kv_bytes_per_token,
        dense_full_kv_bytes_per_query,
        local_window_kv_bytes_per_query,
        candidate_page_kv_bytes_per_query,
        estimated_rt_attention_kv_bytes_per_query,
        estimated_rt_vs_dense_kv_fraction_ppm: fraction_ppm(
            estimated_rt_attention_kv_bytes_per_query,
            dense_full_kv_bytes_per_query,
        ),
        far_oracle_topk_token_kv_bytes_per_query,
        far_oracle_topk_scatter_page_kv_bytes_min_per_query,
        far_oracle_topk_scatter_page_kv_bytes_avg_per_query,
        far_oracle_topk_scatter_page_kv_bytes_max_per_query,
        far_oracle_topk_page_overfetch_avg_x1000: fraction_x1000(
            far_oracle_topk_scatter_page_kv_bytes_avg_per_query,
            far_oracle_topk_token_kv_bytes_per_query,
        ),
        fine_token_projected_candidate_kv_bytes_per_query,
        fine_token_projected_vs_page_candidate_kv_fraction_ppm: fraction_ppm(
            fine_token_projected_candidate_kv_bytes_per_query,
            candidate_page_kv_bytes_per_query,
        ),
        fine_token_learned_projected_candidate_kv_bytes_per_query,
        fine_token_learned_projected_vs_page_candidate_kv_fraction_ppm: fraction_ppm(
            fine_token_learned_projected_candidate_kv_bytes_per_query,
            candidate_page_kv_bytes_per_query,
        ),
    }
}

fn f32_bytes() -> u64 {
    std::mem::size_of::<f32>() as u64
}

fn saturating_mul(lhs: u64, rhs: u64) -> u64 {
    let product = u128::from(lhs) * u128::from(rhs);
    product.min(u128::from(u64::MAX)) as u64
}

fn saturating_mul_div(lhs: u64, rhs: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return 0;
    }
    let product = u128::from(lhs) * u128::from(rhs);
    (product / u128::from(denominator)).min(u128::from(u64::MAX)) as u64
}

fn fraction_ppm(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return 0;
    }
    let scaled = u128::from(numerator) * 1_000_000u128;
    (scaled / u128::from(denominator)).min(u128::from(u64::MAX)) as u64
}

fn fraction_x1000(numerator: u64, denominator: u64) -> u64 {
    if denominator == 0 {
        return 0;
    }
    let scaled = u128::from(numerator) * 1_000u128;
    (scaled / u128::from(denominator)).min(u128::from(u64::MAX)) as u64
}
