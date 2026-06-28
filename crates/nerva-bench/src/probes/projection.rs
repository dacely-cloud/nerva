use nerva_core::types::dtype::DType;
use nerva_runtime::engine::hf_cuda_decode::projection_batch::{
    plan_exact_projection_batch, ProjectionBatchCandidate, ProjectionBatchConfig,
    ProjectionBatchModelKey, ProjectionBatchPlanReason,
};

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
    Ok(projection_batch_plan_json(
        ready_requests,
        compatible_requests,
        config,
        &plan,
    ))
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

fn reason_name(reason: ProjectionBatchPlanReason) -> &'static str {
    match reason {
        ProjectionBatchPlanReason::Ready => "ready",
        ProjectionBatchPlanReason::NoCandidates => "no_candidates",
        ProjectionBatchPlanReason::NoReadyCandidates => "no_ready_candidates",
        ProjectionBatchPlanReason::SharedWeightsUnproven => "shared_weights_unproven",
        ProjectionBatchPlanReason::InsufficientCompatibleReady => "insufficient_compatible_ready",
    }
}
