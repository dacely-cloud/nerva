use nerva_core::types::dtype::DType;
use nerva_runtime::engine::hf_cuda_decode::projection_batch::{
    plan_exact_projection_batch, ProjectionBatchCandidate, ProjectionBatchConfig,
    ProjectionBatchModelKey, ProjectionBatchPlanReason,
};

use crate::json::json_escape;
use crate::parse::parse_optional_u32;

pub(crate) fn run_projection_bench(
    rows: u32,
    cols: u32,
    dtype: u32,
    iterations: u32,
    warmup_iterations: u32,
    block_tokens: u32,
) -> Result<String, String> {
    let summary = nerva_cuda::projection::probe::projection_bench(
        dtype,
        rows,
        cols,
        iterations,
        warmup_iterations,
        block_tokens,
    );
    Ok(summary.to_json())
}

pub(crate) fn run_projection_bench_from_args(args: &[String]) -> Result<String, String> {
    let rows = parse_optional_u32(args.first().cloned(), 64, "rows")?;
    let cols = parse_optional_u32(args.get(1).cloned(), 128, "cols")?;
    let dtype = parse_optional_u32(args.get(2).cloned(), 1, "dtype")?;
    let iterations = parse_optional_u32(args.get(3).cloned(), 16, "iterations")?;
    let warmups = parse_optional_u32(args.get(4).cloned(), 2, "warmup_iterations")?;
    let block_tokens = parse_optional_u32(args.get(5).cloned(), 1, "block_tokens")?;
    run_projection_bench(rows, cols, dtype, iterations, warmups, block_tokens)
}

pub(crate) fn run_projection_batch_plan(
    ready_requests: usize,
    compatible_requests: usize,
    target_block_tokens: usize,
    min_block_tokens: usize,
) -> Result<String, String> {
    let plan_input = projection_batch_plan_input(
        ready_requests,
        compatible_requests,
        target_block_tokens,
        min_block_tokens,
    );
    Ok(projection_batch_plan_json(
        plan_input.ready_requests,
        plan_input.compatible_requests,
        plan_input.config,
        &plan_input.plan,
    ))
}

pub(crate) fn run_projection_batch_exec_probe(
    ready_requests: usize,
    compatible_requests: usize,
    rows: u32,
    cols: u32,
    dtype: u32,
    iterations: u32,
    warmup_iterations: u32,
    target_block_tokens: usize,
    min_block_tokens: usize,
) -> Result<String, String> {
    let plan_input = projection_batch_plan_input(
        ready_requests,
        compatible_requests,
        target_block_tokens,
        min_block_tokens,
    );
    if !plan_input.plan.exact {
        return Ok(projection_batch_exec_probe_skipped_json(
            &plan_input,
            rows,
            cols,
            dtype,
            iterations,
            warmup_iterations,
        ));
    }

    let block_tokens = u32::try_from(plan_input.plan.block_tokens)
        .map_err(|_| "planned block_tokens does not fit in u32".to_string())?;
    let summary = nerva_cuda::projection::probe::projection_bench(
        dtype,
        rows,
        cols,
        iterations,
        warmup_iterations,
        block_tokens,
    );
    Ok(projection_batch_exec_probe_json(&plan_input, &summary))
}

struct ProjectionBatchPlanInput {
    ready_requests: usize,
    compatible_requests: usize,
    config: ProjectionBatchConfig,
    plan: nerva_runtime::engine::hf_cuda_decode::projection_batch::ProjectionBatchPlan,
}

fn projection_batch_plan_input(
    ready_requests: usize,
    compatible_requests: usize,
    target_block_tokens: usize,
    min_block_tokens: usize,
) -> ProjectionBatchPlanInput {
    let compatible_requests = compatible_requests.min(ready_requests);
    let config = ProjectionBatchConfig::new(target_block_tokens, min_block_tokens);
    let candidates = (0..ready_requests)
        .map(|index| ProjectionBatchCandidate {
            request_id: index as u64,
            model: qwen3_8b_model_key(if index < compatible_requests {
                0x7177_656e_335f_3862
            } else {
                0x9000_0000_0000_0000u64.saturating_add(index as u64)
            }),
            prompt_tokens: 128,
            generated_tokens: 16,
            remaining_tokens: 64,
            max_context_tokens: 4096,
            ready: true,
            stopped: false,
        })
        .collect::<Vec<_>>();
    let plan = plan_exact_projection_batch(&candidates, config);
    ProjectionBatchPlanInput {
        ready_requests,
        compatible_requests,
        config,
        plan,
    }
}

fn qwen3_8b_model_key(data_hash: u64) -> ProjectionBatchModelKey {
    ProjectionBatchModelKey {
        data_hash,
        data_hash_available: true,
        dtype: DType::BF16,
        hidden_size: 4096,
        attention_heads: 32,
        kv_heads: 8,
        head_dim: 128,
        intermediate_size: 12288,
        vocab_size: 151936,
        layer_count: 36,
    }
}

fn projection_batch_plan_json(
    ready_requests: usize,
    compatible_requests: usize,
    config: ProjectionBatchConfig,
    plan: &nerva_runtime::engine::hf_cuda_decode::projection_batch::ProjectionBatchPlan,
) -> String {
    let selected = plan
        .selected_request_ids
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let ideal_reuse_x1000 = if plan.exact && plan.block_tokens > 0 {
        plan.block_tokens.saturating_mul(1000)
    } else {
        0
    };
    let ideal_reduction_x100 = if plan.exact && plan.block_tokens > 0 {
        10_000usize.saturating_sub(10_000usize / plan.block_tokens)
    } else {
        0
    };
    format!(
        "{{\"schema\":\"nerva-projection-batch-plan-v1\",\"status\":\"ok\",\"ready_requests\":{},\"compatible_requests\":{},\"target_block_tokens\":{},\"min_block_tokens\":{},\"plan_reason\":\"{}\",\"exact\":{},\"block_tokens\":{},\"selected_request_ids\":[{}],\"ideal_projection_weight_stream_reuse_x1000\":{},\"ideal_projection_weight_stream_reduction_x100\":{},\"requires_shared_weight_hash\":true,\"executor_status\":\"planner_only\"}}",
        ready_requests,
        compatible_requests,
        config.target_block_tokens,
        config.min_block_tokens,
        reason_name(plan.reason),
        plan.exact,
        plan.block_tokens,
        selected,
        ideal_reuse_x1000,
        ideal_reduction_x100,
    )
}

fn projection_batch_exec_probe_skipped_json(
    input: &ProjectionBatchPlanInput,
    rows: u32,
    cols: u32,
    dtype: u32,
    iterations: u32,
    warmup_iterations: u32,
) -> String {
    format!(
        "{{\"schema\":\"nerva-projection-batch-exec-probe-v1\",\"status\":\"skipped\",\"reason\":\"{}\",\"ready_requests\":{},\"compatible_requests\":{},\"target_block_tokens\":{},\"min_block_tokens\":{},\"exact\":false,\"block_tokens\":0,\"rows\":{},\"cols\":{},\"dtype\":{},\"iterations\":{},\"warmup_iterations\":{},\"executor_status\":\"not_executed\"}}",
        reason_name(input.plan.reason),
        input.ready_requests,
        input.compatible_requests,
        input.config.target_block_tokens,
        input.config.min_block_tokens,
        rows,
        cols,
        dtype,
        iterations,
        warmup_iterations,
    )
}

fn projection_batch_exec_probe_json(
    input: &ProjectionBatchPlanInput,
    summary: &nerva_cuda::projection::summary::CudaProjectionBenchSummary,
) -> String {
    let status = if summary.passed() { "ok" } else { "failed" };
    let selected = input
        .plan
        .selected_request_ids
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let block_tokens = usize::try_from(summary.block_tokens).unwrap_or(0);
    let ideal_reuse_x1000 = if input.plan.exact && block_tokens > 0 {
        block_tokens.saturating_mul(1000)
    } else {
        0
    };
    let ideal_reduction_x100 = if input.plan.exact && block_tokens > 0 {
        10_000usize.saturating_sub(10_000usize / block_tokens)
    } else {
        0
    };
    let per_token_speedup_x1000 = if summary.block_cublaslt_graph_speedup_x1000 > 0 {
        summary.block_cublaslt_graph_speedup_x1000
    } else if summary.block_cublaslt_graph_per_token_ns > 0 {
        summary
            .cublaslt_graph_avg_ns
            .saturating_mul(1000)
            .saturating_div(summary.block_cublaslt_graph_per_token_ns)
    } else {
        0
    };
    let error_json = match &summary.error {
        Some(error) => format!("\"{}\"", json_escape(error)),
        None => "null".to_string(),
    };

    format!(
        "{{\"schema\":\"nerva-projection-batch-exec-probe-v1\",\"status\":\"{}\",\"plan_reason\":\"{}\",\"ready_requests\":{},\"compatible_requests\":{},\"target_block_tokens\":{},\"min_block_tokens\":{},\"exact\":{},\"block_tokens\":{},\"selected_request_ids\":[{}],\"rows\":{},\"cols\":{},\"dtype\":{},\"iterations\":{},\"warmup_iterations\":{},\"single_graph_avg_ns\":{},\"block_graph_avg_ns\":{},\"block_graph_per_token_ns\":{},\"block_graph_speedup_x1000\":{},\"ideal_projection_weight_stream_reuse_x1000\":{},\"ideal_projection_weight_stream_reduction_x100\":{},\"mismatch_count\":{},\"max_abs_diff\":{},\"graph_replays\":{},\"graph_captures\":{},\"block_graph_nodes\":{},\"device_allocations\":{},\"device_frees\":{},\"hot_path_allocations\":{},\"executor_status\":\"hardware_block_projection_probe\",\"error\":{}}}",
        status,
        reason_name(input.plan.reason),
        input.ready_requests,
        input.compatible_requests,
        input.config.target_block_tokens,
        input.config.min_block_tokens,
        input.plan.exact,
        summary.block_tokens,
        selected,
        summary.rows,
        summary.cols,
        summary.dtype,
        summary.iterations,
        summary.warmup_iterations,
        summary.cublaslt_graph_avg_ns,
        summary.block_cublaslt_graph_avg_ns,
        summary.block_cublaslt_graph_per_token_ns,
        per_token_speedup_x1000,
        ideal_reuse_x1000,
        ideal_reduction_x100,
        summary.mismatch_count,
        summary.max_abs_diff,
        summary.graph_replays,
        summary.graph_captures,
        summary.block_cublaslt_graph_nodes,
        summary.device_allocations,
        summary.device_frees,
        summary.hot_path_allocations,
        error_json,
    )
}

fn reason_name(reason: ProjectionBatchPlanReason) -> &'static str {
    match reason {
        ProjectionBatchPlanReason::Ready => "ready",
        ProjectionBatchPlanReason::NoCandidates => "no_candidates",
        ProjectionBatchPlanReason::NoReadyCandidates => "no_ready_candidates",
        ProjectionBatchPlanReason::SharedWeightsUnproven => "shared_weights_unproven",
        ProjectionBatchPlanReason::InsufficientCompatibleReady => "insufficient_compatible_ready",
    }
}
