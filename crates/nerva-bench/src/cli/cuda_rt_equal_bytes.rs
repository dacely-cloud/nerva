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

pub(crate) fn fine_token_projected_page_byte_bracket_json(
    summary: &CudaExperimentalRtCandidateBenchSummary,
    sweep: &[CudaExperimentalRtCandidateBenchSummary],
) -> String {
    fine_token_page_byte_bracket_json(
        "synthetic_fine_token_page_byte_bracket",
        summary.fine_token_projected_candidate_tokens,
        summary,
        sweep,
    )
}

pub(crate) fn fine_token_learned_projected_page_byte_bracket_json(
    summary: &CudaExperimentalRtCandidateBenchSummary,
    sweep: &[CudaExperimentalRtCandidateBenchSummary],
) -> String {
    fine_token_page_byte_bracket_json(
        "synthetic_fine_token_learned_page_byte_bracket",
        summary.fine_token_learned_projected_candidate_tokens,
        summary,
        sweep,
    )
}

pub(crate) fn recall_curve_json(sweep: &[CudaExperimentalRtCandidateBenchSummary]) -> String {
    let points = sweep
        .iter()
        .map(recall_curve_point_json)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"scope\":\"recall_vs_candidate_bytes_synthetic\",\"oracle\":\"dense_far_token_topk\",\"quality_axis\":\"far_oracle_topk_token_recall_ppm\",\"bytes_axis\":\"estimated_attention_kv_bytes_per_query\",\"quest_reproduction_status\":\"pending\",\"points\":[{}]}}",
        points,
    )
}

fn recall_curve_point_json(summary: &CudaExperimentalRtCandidateBenchSummary) -> String {
    format!(
        "{{\"candidates_per_query\":{},\"far_oracle\":{},\"selectors\":[{},{},{}],\"norm_sensitivity\":{}}}",
        summary.candidates_per_query,
        far_oracle_curve_json(summary),
        fine_token_selector_curve_json(
            "synthetic_fine_token_projected_3d",
            "synthetic_random_projection_3d",
            summary.fine_token_projected_candidate_tokens,
            summary.fine_token_projected_token_recall_min_ppm,
            summary.fine_token_projected_token_recall_avg_ppm,
            summary,
        ),
        fine_token_selector_curve_json(
            "synthetic_fine_token_learned_projected_4d",
            "synthetic_learned_projection_4d",
            summary.fine_token_learned_projected_candidate_tokens,
            summary.fine_token_learned_projected_token_recall_min_ppm,
            summary.fine_token_learned_projected_token_recall_avg_ppm,
            summary,
        ),
        page_selector_curve_json(summary),
        norm_sensitivity_curve_json(summary),
    )
}

fn far_oracle_curve_json(summary: &CudaExperimentalRtCandidateBenchSummary) -> String {
    let kv_bytes_per_token = kv_bytes_per_token(summary);
    let page_kv_bytes = page_kv_bytes(summary, kv_bytes_per_token);
    let topk_token_kv_bytes = saturating_mul(summary.far_oracle_topk_tokens, kv_bytes_per_token);
    let scatter_avg_page_kv_bytes = saturating_mul_div(
        summary.far_oracle_topk_importance_scatter_avg_pages_x1000,
        page_kv_bytes,
        1_000,
    );
    format!(
        "{{\"topk_tokens\":{},\"topk_token_kv_bytes_per_query\":{},\"scatter_min_pages\":{},\"scatter_avg_pages_x1000\":{},\"scatter_max_pages\":{},\"scatter_page_kv_bytes_avg_per_query\":{},\"scatter_page_overfetch_avg_x1000\":{}}}",
        summary.far_oracle_topk_tokens,
        topk_token_kv_bytes,
        summary.far_oracle_topk_importance_scatter_min_pages,
        summary.far_oracle_topk_importance_scatter_avg_pages_x1000,
        summary.far_oracle_topk_importance_scatter_max_pages,
        scatter_avg_page_kv_bytes,
        fraction_x1000(scatter_avg_page_kv_bytes, topk_token_kv_bytes),
    )
}

fn fine_token_selector_curve_json(
    selector: &str,
    reduction: &str,
    candidate_tokens: u64,
    token_recall_min_ppm: u64,
    token_recall_avg_ppm: u64,
    summary: &CudaExperimentalRtCandidateBenchSummary,
) -> String {
    let kv_bytes_per_token = kv_bytes_per_token(summary);
    let far_kv_bytes = saturating_mul(candidate_tokens, kv_bytes_per_token);
    let total_kv_bytes = local_kv_bytes(summary, kv_bytes_per_token).saturating_add(far_kv_bytes);
    let dense_kv_bytes = dense_kv_bytes(summary, kv_bytes_per_token);
    format!(
        "{{\"selector\":\"{}\",\"reduction\":\"{}\",\"candidate_tokens_per_query\":{},\"far_kv_bytes_per_query\":{},\"estimated_attention_kv_bytes_per_query\":{},\"estimated_vs_dense_kv_fraction_ppm\":{},\"token_recall_min_ppm\":{},\"token_recall_avg_ppm\":{}}}",
        selector,
        reduction,
        candidate_tokens,
        far_kv_bytes,
        total_kv_bytes,
        fraction_ppm(total_kv_bytes, dense_kv_bytes),
        token_recall_min_ppm,
        token_recall_avg_ppm,
    )
}

fn page_selector_curve_json(summary: &CudaExperimentalRtCandidateBenchSummary) -> String {
    let kv_bytes_per_token = kv_bytes_per_token(summary);
    let page_kv_bytes = page_kv_bytes(summary, kv_bytes_per_token);
    let candidate_pages = u64::from(summary.candidates_per_query);
    let far_kv_bytes = saturating_mul(candidate_pages, page_kv_bytes);
    let total_kv_bytes = local_kv_bytes(summary, kv_bytes_per_token).saturating_add(far_kv_bytes);
    let dense_kv_bytes = dense_kv_bytes(summary, kv_bytes_per_token);
    format!(
        "{{\"selector\":\"synthetic_page_level_descriptor_topk\",\"reduction\":\"synthetic_page_descriptor_topk\",\"candidate_pages_per_query\":{},\"far_kv_bytes_per_query\":{},\"estimated_attention_kv_bytes_per_query\":{},\"estimated_vs_dense_kv_fraction_ppm\":{},\"token_recall_min_ppm\":{},\"token_recall_avg_ppm\":{},\"attention_mass_recall_min_ppm\":{},\"attention_mass_recall_avg_ppm\":{}}}",
        candidate_pages,
        far_kv_bytes,
        total_kv_bytes,
        fraction_ppm(total_kv_bytes, dense_kv_bytes),
        summary.page_level_far_oracle_topk_token_recall_min_ppm,
        summary.page_level_far_oracle_topk_token_recall_avg_ppm,
        summary.page_level_attention_mass_recall_min_ppm,
        summary.page_level_attention_mass_recall_avg_ppm,
    )
}

fn norm_sensitivity_curve_json(summary: &CudaExperimentalRtCandidateBenchSummary) -> String {
    format!(
        "{{\"topk_tokens\":{},\"no_augmentation_token_recall_min_ppm\":{},\"no_augmentation_token_recall_avg_ppm\":{},\"synthetic_norm_augmented_token_recall_min_ppm\":{},\"synthetic_norm_augmented_token_recall_avg_ppm\":{},\"synthetic_norm_augmentation_gain_min_ppm\":{},\"synthetic_norm_augmentation_gain_avg_ppm\":{}}}",
        summary.norm_stress_topk_tokens,
        summary.norm_stress_no_augmentation_token_recall_min_ppm,
        summary.norm_stress_no_augmentation_token_recall_avg_ppm,
        summary.norm_stress_synthetic_norm_augmented_token_recall_min_ppm,
        summary.norm_stress_synthetic_norm_augmented_token_recall_avg_ppm,
        signed_delta(
            summary.norm_stress_synthetic_norm_augmented_token_recall_min_ppm,
            summary.norm_stress_no_augmentation_token_recall_min_ppm,
        ),
        signed_delta(
            summary.norm_stress_synthetic_norm_augmented_token_recall_avg_ppm,
            summary.norm_stress_no_augmentation_token_recall_avg_ppm,
        ),
    )
}

fn fine_token_page_byte_bracket_json(
    scope: &str,
    candidate_tokens: u64,
    summary: &CudaExperimentalRtCandidateBenchSummary,
    sweep: &[CudaExperimentalRtCandidateBenchSummary],
) -> String {
    let kv_bytes_per_token = saturating_mul(saturating_mul(u64::from(summary.dims), 2), 4);
    let page_kv_bytes = saturating_mul(u64::from(summary.page_tokens), kv_bytes_per_token);
    let fine_token_far_kv_bytes = saturating_mul(candidate_tokens, kv_bytes_per_token);
    let floor_pages = if page_kv_bytes == 0 {
        0
    } else {
        fine_token_far_kv_bytes / page_kv_bytes
    };
    let ceil_pages = if fine_token_far_kv_bytes == 0 || page_kv_bytes == 0 {
        0
    } else {
        fine_token_far_kv_bytes.div_ceil(page_kv_bytes)
    };

    format!(
        "{{\"scope\":\"{}\",\"selector\":\"synthetic_page_level_descriptor_topk\",\"quest_reproduction_status\":\"pending\",\"fine_token_candidate_tokens_per_query\":{},\"fine_token_far_kv_bytes_per_query\":{},\"page_kv_bytes_per_query\":{},\"floor\":{},\"ceil\":{}}}",
        scope,
        candidate_tokens,
        fine_token_far_kv_bytes,
        page_kv_bytes,
        page_baseline_at_candidate_pages_json(summary, sweep, floor_pages),
        page_baseline_at_candidate_pages_json(summary, sweep, ceil_pages),
    )
}

fn page_baseline_at_candidate_pages_json(
    summary: &CudaExperimentalRtCandidateBenchSummary,
    sweep: &[CudaExperimentalRtCandidateBenchSummary],
    candidate_pages: u64,
) -> String {
    let baseline = u32::try_from(candidate_pages).ok().and_then(|pages| {
        sweep
            .iter()
            .find(|point| point.candidates_per_query == pages)
    });
    let kv_bytes_per_token = saturating_mul(saturating_mul(u64::from(summary.dims), 2), 4);
    let page_kv_bytes = saturating_mul(u64::from(summary.page_tokens), kv_bytes_per_token);
    let local_kv_bytes = saturating_mul(summary.local_window_tokens, kv_bytes_per_token);
    let far_kv_bytes = saturating_mul(candidate_pages, page_kv_bytes);
    let total_kv_bytes = local_kv_bytes.saturating_add(far_kv_bytes);
    let dense_kv_bytes = saturating_mul(
        saturating_mul(u64::from(summary.pages), u64::from(summary.page_tokens)),
        kv_bytes_per_token,
    );
    let (
        measured,
        attention_mass_recall_min_ppm,
        attention_mass_recall_avg_ppm,
        far_oracle_topk_token_recall_min_ppm,
        far_oracle_topk_token_recall_avg_ppm,
    ) = match baseline {
        Some(point) => (
            true,
            point.page_level_attention_mass_recall_min_ppm,
            point.page_level_attention_mass_recall_avg_ppm,
            point.page_level_far_oracle_topk_token_recall_min_ppm,
            point.page_level_far_oracle_topk_token_recall_avg_ppm,
        ),
        None => (false, 0, 0, 0, 0),
    };

    format!(
        "{{\"candidate_pages_per_query\":{},\"far_kv_bytes_per_query\":{},\"estimated_attention_kv_bytes_per_query\":{},\"estimated_vs_dense_kv_fraction_ppm\":{},\"measured\":{},\"attention_mass_recall_min_ppm\":{},\"attention_mass_recall_avg_ppm\":{},\"far_oracle_topk_token_recall_min_ppm\":{},\"far_oracle_topk_token_recall_avg_ppm\":{}}}",
        candidate_pages,
        far_kv_bytes,
        total_kv_bytes,
        fraction_ppm(total_kv_bytes, dense_kv_bytes),
        measured,
        attention_mass_recall_min_ppm,
        attention_mass_recall_avg_ppm,
        far_oracle_topk_token_recall_min_ppm,
        far_oracle_topk_token_recall_avg_ppm,
    )
}

fn kv_bytes_per_token(summary: &CudaExperimentalRtCandidateBenchSummary) -> u64 {
    saturating_mul(saturating_mul(u64::from(summary.dims), 2), 4)
}

fn page_kv_bytes(
    summary: &CudaExperimentalRtCandidateBenchSummary,
    kv_bytes_per_token: u64,
) -> u64 {
    saturating_mul(u64::from(summary.page_tokens), kv_bytes_per_token)
}

fn local_kv_bytes(
    summary: &CudaExperimentalRtCandidateBenchSummary,
    kv_bytes_per_token: u64,
) -> u64 {
    saturating_mul(summary.local_window_tokens, kv_bytes_per_token)
}

fn dense_kv_bytes(
    summary: &CudaExperimentalRtCandidateBenchSummary,
    kv_bytes_per_token: u64,
) -> u64 {
    saturating_mul(
        saturating_mul(u64::from(summary.pages), u64::from(summary.page_tokens)),
        kv_bytes_per_token,
    )
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

fn signed_delta(lhs: u64, rhs: u64) -> i128 {
    i128::from(lhs) - i128::from(rhs)
}
