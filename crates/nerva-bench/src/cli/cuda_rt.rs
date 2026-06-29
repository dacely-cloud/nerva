use std::process::ExitCode;

use nerva_cuda::experimental_rt::probe::experimental_rt_candidate_bench;
use nerva_cuda::experimental_rt::summary::CudaExperimentalRtCandidateBenchSummary;

use crate::cli::cuda_rt_equal_bytes::{
    fine_token_learned_projected_page_byte_bracket_json,
    fine_token_projected_page_byte_bracket_json, page_level_equal_bytes_baseline_json,
    recall_curve_json,
};
use crate::parse::{parse_optional_u32, parse_optional_usize};

const DIMS: u32 = 16;
const WARMUP_ITERATIONS: u32 = 8;
const DEFAULT_LAYER_COUNT: u32 = 36;
const DEFAULT_MATRIX_ITERATIONS: u32 = 16;
const MATRIX_CONTEXT_TOKENS: [usize; 4] = [128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024];
const MATRIX_QUERY_COUNTS: [u32; 3] = [1, 8, 32];
const MATRIX_CANDIDATE_PAGES: [u32; 4] = [128, 256, 512, 1024];

#[derive(Clone, Copy)]
pub(crate) struct RtArgs {
    pub(crate) context_tokens: usize,
    pub(crate) query_count: u32,
    pub(crate) candidates_per_query: u32,
    pub(crate) iterations: u32,
    pub(crate) page_tokens: u32,
    pub(crate) layer_count: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct RtMatrixConfig {
    pub(crate) iterations: u32,
    pub(crate) page_tokens: u32,
    pub(crate) layer_count: u32,
}

pub(crate) struct RtMatrixPoint {
    pub(crate) context_tokens: usize,
    pub(crate) query_count: u32,
    pub(crate) candidates_per_query: u32,
    pub(crate) summary: CudaExperimentalRtCandidateBenchSummary,
}

pub(crate) fn run_experimental_rt(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let config = match parse_rt_args(args, 128, 128) {
        Ok(mut config) => {
            config.layer_count =
                match parse_optional_u32(args.next(), DEFAULT_LAYER_COUNT, "layer_count") {
                    Ok(layer_count) => layer_count.max(1),
                    Err(reason) => return parse_error(reason),
                };
            config
        }
        Err(reason) => return parse_error(reason),
    };
    let pages = pages_for_context(config.context_tokens, config.page_tokens);
    let summary = run_candidate_point(&config, config.candidates_per_query.min(pages).max(1));
    print_status_json(
        single_json(config.context_tokens, config.layer_count, &summary),
        summary.passed(),
    )
}

pub(crate) fn run_experimental_rt_sweep(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let mut config = match parse_rt_args(args, 128, 64) {
        Ok(config) => config,
        Err(reason) => return parse_error(reason),
    };
    let min_candidates = match parse_optional_u32(args.next(), 1, "min_candidates_per_query") {
        Ok(min_candidates) => min_candidates,
        Err(reason) => return parse_error(reason),
    };
    config.layer_count = match parse_optional_u32(args.next(), DEFAULT_LAYER_COUNT, "layer_count") {
        Ok(layer_count) => layer_count.max(1),
        Err(reason) => return parse_error(reason),
    };
    let pages = pages_for_context(config.context_tokens, config.page_tokens);
    let sizes = candidate_sweep_values(pages, min_candidates, config.candidates_per_query);
    let summaries = sizes
        .iter()
        .map(|candidates| run_candidate_point(&config, *candidates))
        .collect::<Vec<_>>();
    let passed = summaries
        .iter()
        .all(CudaExperimentalRtCandidateBenchSummary::passed);
    print_status_json(sweep_json(&config, &summaries), passed)
}

pub(crate) fn run_experimental_rt_matrix(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let config = match parse_matrix_args(args) {
        Ok(config) => config,
        Err(reason) => return parse_error(reason),
    };
    let mut points = Vec::new();
    for context_tokens in MATRIX_CONTEXT_TOKENS {
        for query_count in MATRIX_QUERY_COUNTS {
            for candidates in MATRIX_CANDIDATE_PAGES {
                let rt_args = RtArgs {
                    context_tokens,
                    query_count,
                    candidates_per_query: candidates,
                    iterations: config.iterations,
                    page_tokens: config.page_tokens,
                    layer_count: config.layer_count,
                };
                let pages = pages_for_context(context_tokens, config.page_tokens);
                let candidates_per_query = candidates.min(pages).max(1);
                let summary = run_candidate_point(&rt_args, candidates_per_query);
                points.push(RtMatrixPoint {
                    context_tokens,
                    query_count,
                    candidates_per_query,
                    summary,
                });
            }
        }
    }
    let passed = points.iter().all(|point| point.summary.passed());
    print_status_json(matrix_json(&config, &points), passed)
}

fn parse_rt_args(
    args: &mut impl Iterator<Item = String>,
    default_candidates: u32,
    default_iterations: u32,
) -> Result<RtArgs, String> {
    Ok(RtArgs {
        context_tokens: parse_optional_usize(args.next(), 128 * 1024, "context_tokens")?,
        query_count: parse_optional_u32(args.next(), 1, "query_count")?,
        candidates_per_query: parse_optional_u32(
            args.next(),
            default_candidates,
            "candidates_per_query",
        )?,
        iterations: parse_optional_u32(args.next(), default_iterations, "iterations")?,
        page_tokens: parse_optional_u32(args.next(), 64, "page_tokens")?.max(1),
        layer_count: DEFAULT_LAYER_COUNT,
    })
}

fn parse_matrix_args(args: &mut impl Iterator<Item = String>) -> Result<RtMatrixConfig, String> {
    Ok(RtMatrixConfig {
        iterations: parse_optional_u32(args.next(), DEFAULT_MATRIX_ITERATIONS, "iterations")?,
        page_tokens: parse_optional_u32(args.next(), 64, "page_tokens")?.max(1),
        layer_count: parse_optional_u32(args.next(), DEFAULT_LAYER_COUNT, "layer_count")?.max(1),
    })
}

fn run_candidate_point(
    config: &RtArgs,
    candidates_per_query: u32,
) -> CudaExperimentalRtCandidateBenchSummary {
    experimental_rt_candidate_bench(
        pages_for_context(config.context_tokens, config.page_tokens),
        config.page_tokens,
        DIMS,
        config.query_count,
        candidates_per_query,
        config.iterations,
        WARMUP_ITERATIONS,
    )
}

fn pages_for_context(context_tokens: usize, page_tokens: u32) -> u32 {
    context_tokens
        .saturating_add(page_tokens as usize - 1)
        .saturating_div(page_tokens as usize)
        .max(1)
        .min(u32::MAX as usize) as u32
}

pub(crate) fn candidate_sweep_values(
    pages: u32,
    min_candidates: u32,
    max_candidates: u32,
) -> Vec<u32> {
    let upper = max_candidates.min(pages).max(1);
    let lower = min_candidates.min(upper).max(1);
    let mut values = vec![lower];
    let mut candidate = 1u32;
    while candidate < lower && candidate <= u32::MAX / 2 {
        candidate *= 2;
    }
    if candidate == lower && candidate <= u32::MAX / 2 {
        candidate *= 2;
    }
    while candidate < upper {
        values.push(candidate);
        if candidate > u32::MAX / 2 {
            break;
        }
        candidate *= 2;
    }
    if values.last().copied() != Some(upper) {
        values.push(upper);
    }
    values.sort_unstable();
    values.dedup();
    values
}

fn single_json(
    context_tokens: usize,
    layer_count: u32,
    summary: &CudaExperimentalRtCandidateBenchSummary,
) -> String {
    format!(
        "{{\"status\":\"{}\",\"backend\":\"cuda\",\"mode\":\"experimental_rt_candidate\",\"scope\":\"attention_stage_synthetic\",\"context_tokens\":{},\"layer_count\":{},\"summary\":{},\"decode_latency_estimate\":{}}}",
        if summary.passed() { "ok" } else { "failed" },
        context_tokens,
        layer_count,
        summary.to_json(),
        decode_latency_estimate_json(summary, layer_count),
    )
}

pub(crate) fn sweep_json(
    config: &RtArgs,
    summaries: &[CudaExperimentalRtCandidateBenchSummary],
) -> String {
    let points = summaries
        .iter()
        .map(|summary| {
            format!(
                "{{\"candidates_per_query\":{},\"summary\":{},\"page_level_equal_bytes_baseline\":{},\"fine_token_projected_page_byte_bracket\":{},\"fine_token_learned_projected_page_byte_bracket\":{},\"decode_latency_estimate\":{}}}",
                summary.candidates_per_query,
                summary.to_json(),
                page_level_equal_bytes_baseline_json(summary),
                fine_token_projected_page_byte_bracket_json(summary, summaries),
                fine_token_learned_projected_page_byte_bracket_json(summary, summaries),
                decode_latency_estimate_json(summary, config.layer_count),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let passed = summaries
        .iter()
        .all(CudaExperimentalRtCandidateBenchSummary::passed);
    format!(
        "{{\"status\":\"{}\",\"backend\":\"cuda\",\"mode\":\"experimental_rt_candidate_sweep\",\"scope\":\"recall_candidate_size_synthetic\",\"context_tokens\":{},\"page_tokens\":{},\"dims\":{},\"query_count\":{},\"iterations\":{},\"warmup_iterations\":{},\"layer_count\":{},\"recall_curve\":{},\"points\":[{}]}}",
        if passed { "ok" } else { "failed" },
        config.context_tokens,
        config.page_tokens,
        DIMS,
        config.query_count,
        config.iterations,
        WARMUP_ITERATIONS,
        config.layer_count,
        recall_curve_json(summaries),
        points,
    )
}

pub(crate) fn matrix_json(config: &RtMatrixConfig, points: &[RtMatrixPoint]) -> String {
    let point_json = points
        .iter()
        .map(|point| {
            format!(
                "{{\"context_tokens\":{},\"query_count\":{},\"candidates_per_query\":{},\"summary\":{},\"page_level_equal_bytes_baseline\":{},\"decode_latency_estimate\":{}}}",
                point.context_tokens,
                point.query_count,
                point.candidates_per_query,
                point.summary.to_json(),
                page_level_equal_bytes_baseline_json(&point.summary),
                decode_latency_estimate_json(&point.summary, config.layer_count),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let passed = points.iter().all(|point| point.summary.passed());
    format!(
        "{{\"status\":\"{}\",\"backend\":\"cuda\",\"mode\":\"experimental_rt_matrix\",\"scope\":\"recall_context_candidate_synthetic_matrix\",\"context_tokens\":[{}],\"query_counts\":[{}],\"candidate_pages\":[{}],\"page_tokens\":{},\"dims\":{},\"iterations\":{},\"warmup_iterations\":{},\"layer_count\":{},\"recall_curves\":[{}],\"points\":[{}]}}",
        if passed { "ok" } else { "failed" },
        join_usize(MATRIX_CONTEXT_TOKENS),
        join_u32(MATRIX_QUERY_COUNTS),
        join_u32(MATRIX_CANDIDATE_PAGES),
        config.page_tokens,
        DIMS,
        config.iterations,
        WARMUP_ITERATIONS,
        config.layer_count,
        matrix_recall_curves_json(points),
        point_json,
    )
}

fn matrix_recall_curves_json(points: &[RtMatrixPoint]) -> String {
    let mut groups = Vec::new();
    let mut seen = Vec::new();
    for point in points {
        let key = (point.context_tokens, point.query_count);
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let summaries = points
            .iter()
            .filter(|candidate| {
                candidate.context_tokens == point.context_tokens
                    && candidate.query_count == point.query_count
            })
            .map(|candidate| candidate.summary.clone())
            .collect::<Vec<_>>();
        groups.push(format!(
            "{{\"context_tokens\":{},\"query_count\":{},\"recall_curve\":{}}}",
            point.context_tokens,
            point.query_count,
            recall_curve_json(&summaries),
        ));
    }
    groups.join(",")
}

pub(crate) fn decode_latency_estimate_json(
    summary: &CudaExperimentalRtCandidateBenchSummary,
    layer_count: u32,
) -> String {
    let layers = u64::from(layer_count.max(1));
    let dense_selector_stage = summary.dense_selector_attention_stage_avg_ns;
    let rt_stage = summary.rt_selector_attention_stage_avg_ns;
    let rt_overlapped_stage = summary.rt_selector_overlapped_attention_stage_avg_ns;
    let dense_full_stage = summary.dense_full_attention_avg_ns;
    let rt_saved = dense_selector_stage.saturating_sub(rt_stage);
    let rt_overlapped_saved = dense_selector_stage.saturating_sub(rt_overlapped_stage);
    let dense_full_rt_overlapped_saved = dense_full_stage.saturating_sub(rt_overlapped_stage);
    let dense_selector_token = dense_selector_stage.saturating_mul(layers);
    let rt_token = rt_stage.saturating_mul(layers);
    let rt_overlapped_token = rt_overlapped_stage.saturating_mul(layers);
    let dense_full_token = dense_full_stage.saturating_mul(layers);
    let rt_saved_token = rt_saved.saturating_mul(layers);
    let rt_overlapped_saved_token = rt_overlapped_saved.saturating_mul(layers);
    let dense_full_rt_overlapped_saved_token =
        dense_full_rt_overlapped_saved.saturating_mul(layers);
    format!(
        "{{\"scope\":\"attention_stage_derived\",\"full_decode_latency_measured\":false,\"note\":\"Derived from measured attention-stage timings multiplied by layer_count; excludes projection, MLP, sampler, host/runtime overhead, and quality effects.\",\"layer_count\":{},\"dense_selector_path_ns_per_layer\":{},\"rt_selector_path_ns_per_layer\":{},\"rt_overlapped_path_ns_per_layer\":{},\"dense_full_attention_ns_per_layer\":{},\"rt_vs_dense_selector_saved_ns_per_layer\":{},\"rt_overlapped_vs_dense_selector_saved_ns_per_layer\":{},\"rt_overlapped_vs_dense_full_saved_ns_per_layer\":{},\"dense_selector_path_estimated_ns_per_token\":{},\"rt_selector_path_estimated_ns_per_token\":{},\"rt_overlapped_path_estimated_ns_per_token\":{},\"dense_full_attention_estimated_ns_per_token\":{},\"rt_vs_dense_selector_saved_ns_per_token\":{},\"rt_overlapped_vs_dense_selector_saved_ns_per_token\":{},\"rt_overlapped_vs_dense_full_saved_ns_per_token\":{},\"rt_vs_dense_selector_saved_ms_per_token\":{:.6},\"rt_overlapped_vs_dense_selector_saved_ms_per_token\":{:.6},\"rt_overlapped_vs_dense_full_saved_ms_per_token\":{:.6}}}",
        layer_count.max(1),
        dense_selector_stage,
        rt_stage,
        rt_overlapped_stage,
        dense_full_stage,
        rt_saved,
        rt_overlapped_saved,
        dense_full_rt_overlapped_saved,
        dense_selector_token,
        rt_token,
        rt_overlapped_token,
        dense_full_token,
        rt_saved_token,
        rt_overlapped_saved_token,
        dense_full_rt_overlapped_saved_token,
        ns_to_ms(rt_saved_token),
        ns_to_ms(rt_overlapped_saved_token),
        ns_to_ms(dense_full_rt_overlapped_saved_token),
    )
}

fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn join_u32<const N: usize>(values: [u32; N]) -> String {
    values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn join_usize<const N: usize>(values: [usize; N]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_error(reason: String) -> ExitCode {
    eprintln!("{reason}");
    ExitCode::from(2)
}

fn print_status_json(json: String, passed: bool) -> ExitCode {
    println!("{json}");
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
