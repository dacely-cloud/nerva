use std::time::Instant;

use nerva_core::types::dtype::DType;
use nerva_cuda::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use nerva_cuda::decode::hf_sequence::request::CUDA_HF_DECODE_SEQUENCE_DTYPE_F16;
use nerva_cuda::decode::hf_sequence::session::request::{
    CudaHfDecodeSequenceSession, CudaHfDecodeSequenceSessionConfig,
};
use nerva_cuda::decode::hf_sequence::session::stateful::CudaHfDecodeSequenceLoop;
use nerva_cuda::decode::hf_sequence::weight_plan::{
    CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT, CudaHfDecodeSequenceWeightBlock,
    CudaHfDecodeSequenceWeightPlan, hash_weight_blocks,
};
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_runtime::engine::hf_cuda_decode::continuous_batch::{
    CudaDecodeLoopBatchEntry, advance_continuous_decode_batch_once,
};
use nerva_runtime::engine::hf_cuda_decode::projection_batch::{
    ProjectionBatchCandidate, ProjectionBatchConfig, ProjectionBatchModelKey,
    ProjectionBatchPlanReason, plan_exact_projection_batch,
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

pub(crate) fn run_projection_batch_advance_probe(
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
    if !plan_input.plan.exact {
        return Ok(projection_batch_advance_probe_skipped_json(&plan_input));
    }

    let block_tokens = plan_input.plan.block_tokens;
    let mut sequential_sessions = create_synthetic_batch_sessions(block_tokens)?;
    let mut batch_sessions = create_synthetic_batch_sessions(block_tokens)?;
    let sequential = run_synthetic_sequential_advance(&mut sequential_sessions)?;
    let batch =
        run_synthetic_batched_advance(&mut batch_sessions, target_block_tokens, min_block_tokens)?;
    Ok(projection_batch_advance_probe_json(
        &plan_input,
        &sequential,
        &batch,
    ))
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

struct SequentialAdvanceProbe {
    wall_ns: u128,
    tokens: Vec<u32>,
    ok: bool,
}

struct BatchedAdvanceProbe {
    wall_ns: u128,
    summary:
        nerva_cuda::decode::hf_sequence::session::request::CudaHfDecodeSequenceBatchAdvanceSummary,
}

fn run_synthetic_sequential_advance(
    sessions: &mut [CudaHfDecodeSequenceSession],
) -> Result<SequentialAdvanceProbe, String> {
    let mut loops = start_and_drain_first_tokens(sessions)?;
    let started = Instant::now();
    let mut tokens = Vec::with_capacity(loops.len());
    let mut ok = true;
    for loop_state in &mut loops {
        let summary = loop_state.advance(1);
        ok = ok && summary.status == SmokeStatus::Ok && summary.tokens.len() == 1;
        tokens.extend(summary.tokens.iter().copied());
    }
    Ok(SequentialAdvanceProbe {
        wall_ns: started.elapsed().as_nanos(),
        tokens,
        ok,
    })
}

fn run_synthetic_batched_advance(
    sessions: &mut [CudaHfDecodeSequenceSession],
    target_block_tokens: usize,
    min_block_tokens: usize,
) -> Result<BatchedAdvanceProbe, String> {
    let mut loops = start_and_drain_first_tokens(sessions)?;
    let entries = loops
        .iter_mut()
        .enumerate()
        .map(|(index, loop_state)| CudaDecodeLoopBatchEntry {
            candidate: synthetic_projection_batch_candidate(index as u64),
            loop_state,
        })
        .collect::<Vec<_>>();
    let started = Instant::now();
    let output = advance_continuous_decode_batch_once(
        entries,
        ProjectionBatchConfig::new(target_block_tokens, min_block_tokens),
    );
    if !output.used_batched_projection() {
        return Err(format!(
            "continuous batch scheduler did not use batched projection: {:?}",
            output.records
        ));
    }
    let summary = output
        .selected
        .and_then(|selected| selected.batch)
        .ok_or_else(|| "continuous batch scheduler returned no batch summary".to_string())?;
    if summary.target_block_tokens
        != u32::try_from(target_block_tokens)
            .map_err(|_| "target_block_tokens does not fit in u32".to_string())?
        || summary.min_block_tokens
            != u32::try_from(min_block_tokens)
                .map_err(|_| "min_block_tokens does not fit in u32".to_string())?
    {
        return Err("continuous batch scheduler returned unexpected block config".to_string());
    }
    Ok(BatchedAdvanceProbe {
        wall_ns: started.elapsed().as_nanos(),
        summary,
    })
}

fn synthetic_projection_batch_candidate(request_id: u64) -> ProjectionBatchCandidate {
    ProjectionBatchCandidate {
        request_id,
        model: ProjectionBatchModelKey {
            data_hash: 0x7379_6e74_6865_7469,
            data_hash_available: true,
            dtype: DType::F16,
            hidden_size: 128,
            attention_heads: 2,
            kv_heads: 2,
            head_dim: 64,
            intermediate_size: 256,
            vocab_size: 8,
            layer_count: 1,
        },
        prompt_tokens: 1,
        generated_tokens: 1,
        remaining_tokens: 1,
        max_context_tokens: 5,
        ready: true,
        stopped: false,
    }
}

fn start_and_drain_first_tokens(
    sessions: &mut [CudaHfDecodeSequenceSession],
) -> Result<Vec<CudaHfDecodeSequenceLoop<'_>>, String> {
    let mut loops = Vec::with_capacity(sessions.len());
    for (index, session) in sessions.iter_mut().enumerate() {
        let prompt = [u32::try_from(index % 2).unwrap_or(0)];
        let started = CudaHfDecodeSequenceLoop::start(session, &prompt, None);
        if started.summary.status != SmokeStatus::Ok {
            return Err(started
                .summary
                .error
                .unwrap_or_else(|| "CUDA synthetic session start failed".to_string()));
        }
        let mut loop_state = started.loop_state.unwrap();
        let first = loop_state.advance(1);
        if first.status != SmokeStatus::Ok || first.tokens.len() != 1 {
            return Err(first
                .error
                .unwrap_or_else(|| "CUDA synthetic first token drain failed".to_string()));
        }
        loops.push(loop_state);
    }
    Ok(loops)
}

fn create_synthetic_batch_sessions(
    count: usize,
) -> Result<Vec<CudaHfDecodeSequenceSession>, String> {
    let one = 0x3c00;
    let zero = 0x0000;
    let hidden = 128usize;
    let intermediate = 256usize;
    let vocab_size = 8usize;
    let embeddings = vec![zero; vocab_size * hidden];
    let rms = vec![one; hidden];
    let attn_matrix = vec![zero; hidden * hidden];
    let mlp_matrix = vec![zero; intermediate * hidden];
    let down_matrix = vec![zero; hidden * intermediate];
    let lm_head = vec![zero; vocab_size * hidden];
    let layer = CudaHfDecodeChainLayer {
        rms_attn_weight: &rms,
        rms_mlp_weight: &rms,
        w_q: &attn_matrix,
        w_k: &attn_matrix,
        q_norm_weight: None,
        k_norm_weight: None,
        w_v: &attn_matrix,
        w_o: &attn_matrix,
        q_bias: None,
        k_bias: None,
        v_bias: None,
        o_bias: None,
        w_gate: &mlp_matrix,
        w_up: &mlp_matrix,
        w_down: &down_matrix,
    };
    let layers = [layer];
    let weight_blocks = synthetic_weight_blocks(
        &embeddings,
        &rms,
        &attn_matrix,
        &mlp_matrix,
        &down_matrix,
        &lm_head,
    );
    let weight_plan = synthetic_weight_plan(&weight_blocks);
    let config = CudaHfDecodeSequenceSessionConfig {
        dtype: CUDA_HF_DECODE_SEQUENCE_DTYPE_F16,
        hidden,
        heads: 1,
        kv_heads: 1,
        head_dim: hidden,
        intermediate,
        vocab_size,
        max_context_tokens: 5,
        rms_eps: 1e-5,
        rope_theta: None,
        embeddings: &embeddings,
        layers: &layers,
        final_norm_weight: &rms,
        lm_head: &lm_head,
        weight_plan: Some(weight_plan),
        weight_blocks: &weight_blocks,
        detailed_profile: false,
    };
    let mut sessions = Vec::with_capacity(count);
    if count == 0 {
        return Ok(sessions);
    }
    let created = config.create();
    if created.summary.status != SmokeStatus::Ok {
        return Err(created
            .summary
            .error
            .unwrap_or_else(|| "CUDA synthetic session create failed".to_string()));
    }
    let mut parent = created.session.unwrap();
    for _ in 1..count {
        let forked = parent.fork_shared_weights(false);
        if forked.summary.status != SmokeStatus::Ok {
            return Err(forked
                .summary
                .error
                .unwrap_or_else(|| "CUDA synthetic shared-weight fork failed".to_string()));
        }
        sessions.push(forked.session.unwrap());
    }
    sessions.insert(0, parent);
    Ok(sessions)
}

fn synthetic_weight_plan(
    descriptors: &[CudaHfDecodeSequenceWeightBlock],
) -> CudaHfDecodeSequenceWeightPlan {
    let weight_bytes = descriptors
        .iter()
        .map(|descriptor| descriptor.bytes)
        .sum::<u64>();
    CudaHfDecodeSequenceWeightPlan {
        blocks: descriptors.len() as u32,
        gpu_resident_blocks: descriptors.len() as u32,
        gpu_staged_blocks: 0,
        weight_bytes,
        gpu_resident_weight_bytes: weight_bytes,
        gpu_staged_weight_bytes: 0,
        descriptor_hash: hash_weight_blocks(descriptors),
    }
}

fn synthetic_weight_blocks(
    embeddings: &[u16],
    rms: &[u16],
    attn_matrix: &[u16],
    mlp_matrix: &[u16],
    down_matrix: &[u16],
    lm_head: &[u16],
) -> Vec<CudaHfDecodeSequenceWeightBlock> {
    let mut offset = 0u64;
    let mut block_id = 0u64;
    let mut blocks = Vec::new();
    for source in [
        embeddings,
        rms,
        rms,
        attn_matrix,
        attn_matrix,
        attn_matrix,
        attn_matrix,
        mlp_matrix,
        mlp_matrix,
        down_matrix,
        rms,
        lm_head,
    ] {
        let bytes = (source.len() * std::mem::size_of::<u16>()) as u64;
        blocks.push(CudaHfDecodeSequenceWeightBlock {
            host_source: source.as_ptr(),
            source_file: std::ptr::null(),
            source_file_len: 0,
            file_offset_begin: 0,
            block_id,
            block_version: 1,
            offset_bytes: offset,
            bytes,
            strategy: CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT,
            reserved: 0,
        });
        offset = offset.saturating_add(bytes);
        block_id = block_id.saturating_add(1);
    }
    blocks
}

fn projection_batch_advance_probe_skipped_json(input: &ProjectionBatchPlanInput) -> String {
    format!(
        "{{\"schema\":\"nerva-projection-batch-advance-probe-v1\",\"status\":\"skipped\",\"reason\":\"{}\",\"ready_requests\":{},\"compatible_requests\":{},\"target_block_tokens\":{},\"min_block_tokens\":{},\"exact\":false,\"block_tokens\":0,\"executor_status\":\"not_executed\"}}",
        reason_name(input.plan.reason),
        input.ready_requests,
        input.compatible_requests,
        input.config.target_block_tokens,
        input.config.min_block_tokens,
    )
}

fn projection_batch_advance_probe_json(
    input: &ProjectionBatchPlanInput,
    sequential: &SequentialAdvanceProbe,
    batch: &BatchedAdvanceProbe,
) -> String {
    let batch_summary = &batch.summary;
    let status = if sequential.ok && batch_summary.status == SmokeStatus::Ok {
        "ok"
    } else {
        "failed"
    };
    let wall_speedup_x1000 = if batch.wall_ns > 0 {
        sequential
            .wall_ns
            .saturating_mul(1000)
            .saturating_div(batch.wall_ns)
    } else {
        0
    };
    format!(
        "{{\"schema\":\"nerva-projection-batch-advance-probe-v1\",\"status\":\"{}\",\"plan_reason\":\"{}\",\"ready_requests\":{},\"compatible_requests\":{},\"target_block_tokens\":{},\"min_block_tokens\":{},\"shared_weight_forks\":true,\"exact\":{},\"block_tokens\":{},\"observed_tokens\":{},\"sequential_wall_ns\":{},\"batch_wall_ns\":{},\"sequential_vs_batch_wall_speedup_x1000\":{},\"batch_projection_elapsed_ns\":{},\"projection_kernel_launches\":{},\"pack_kernel_launches\":{},\"scatter_kernel_launches\":{},\"dependency_kernel_launches\":{},\"sampling_kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"sequential_tokens\":{:?},\"batch_tokens\":{:?},\"executor_status\":\"continuous_runtime_batch_scheduler\"}}",
        status,
        reason_name(input.plan.reason),
        input.ready_requests,
        input.compatible_requests,
        input.config.target_block_tokens,
        input.config.min_block_tokens,
        batch_summary.exact,
        batch_summary.block_tokens,
        batch_summary.observed_tokens,
        sequential.wall_ns,
        batch.wall_ns,
        wall_speedup_x1000,
        batch_summary.projection_elapsed_ns,
        batch_summary.projection_kernel_launches,
        batch_summary.pack_kernel_launches,
        batch_summary.scatter_kernel_launches,
        batch_summary.dependency_kernel_launches,
        batch_summary.sampling_kernel_launches,
        batch_summary.sync_calls,
        batch_summary.hot_path_allocations,
        sequential.tokens,
        batch_summary.tokens,
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
