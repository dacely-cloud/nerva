use std::fs;
use std::process::ExitCode;

use nerva_core::types::id::token::TokenId;
use nerva_cuda::experimental_rt::probe::experimental_rt_candidate_bench;
use nerva_cuda::experimental_rt::summary::CudaExperimentalRtCandidateBenchSummary;
use nerva_model::hf::tokenizer::encode_text_prompt;
use nerva_runtime::engine::hf_cuda_decode::file_backed::shared_fork_batch::{
    run_hf_causal_lm_cuda_shared_fork_batch_probe, HfCudaSharedForkBatchOutput,
    HfCudaSharedForkBatchSchedulerSummary,
};
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

use crate::cli::exit;
use crate::cli::model::causal_lm_cuda_session_stream::u32s_json;
use crate::json::json_escape;
use crate::parse::{parse_optional_u32, parse_optional_usize};

pub(crate) const SHARED_FORK_STORY_PROMPT: &str = "Write me a long story about a clockwork city under the ocean. Include vivid sensory detail, dialogue, conflict, and a clear ending.";

pub(crate) fn run_hf_causal_lm_cuda_shared_fork_batch(
    args: &mut impl Iterator<Item = String>,
) -> ExitCode {
    let (items, experimental_rt) = strip_experimental_rt_arg(args);
    let mut args = items.into_iter();
    let path = args.next();
    let request_count = match parse_optional_usize(args.next(), 2, "request_count") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let max_context = match parse_optional_usize(args.next(), 8192, "max_context_tokens") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let max_new_tokens = match parse_optional_usize(args.next(), 128, "max_new_tokens") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let target_block_tokens =
        match parse_optional_usize(args.next(), request_count, "target_block_tokens") {
            Ok(value) => value,
            Err(reason) => return exit::parse_error(reason),
        };
    let min_block_tokens = match parse_optional_usize(args.next(), 2, "min_block_tokens") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let prompt_spec = args.next();
    let compute_capability = match parse_optional_u32(args.next(), 0, "compute_capability") {
        Ok(0) => None,
        Ok(value) => Some(value),
        Err(reason) => return exit::parse_error(reason),
    };
    exit::print_json_result(hf_causal_lm_cuda_shared_fork_batch_json(
        path,
        request_count,
        max_context,
        max_new_tokens,
        target_block_tokens,
        min_block_tokens,
        prompt_spec,
        compute_capability,
        experimental_rt,
    ))
}

pub(crate) fn run_hf_causal_lm_cuda_shared_fork_batch_compare(
    args: &mut impl Iterator<Item = String>,
) -> ExitCode {
    let (items, experimental_rt) = strip_experimental_rt_arg(args);
    let mut args = items.into_iter();
    let path = args.next();
    let request_count = match parse_optional_usize(args.next(), 2, "request_count") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let max_context = match parse_optional_usize(args.next(), 8192, "max_context_tokens") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let max_new_tokens = match parse_optional_usize(args.next(), 128, "max_new_tokens") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let target_block_tokens =
        match parse_optional_usize(args.next(), request_count, "target_block_tokens") {
            Ok(value) => value,
            Err(reason) => return exit::parse_error(reason),
        };
    let min_block_tokens = match parse_optional_usize(args.next(), 2, "min_block_tokens") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let prompt_spec = args.next();
    let compute_capability = match parse_optional_u32(args.next(), 0, "compute_capability") {
        Ok(0) => None,
        Ok(value) => Some(value),
        Err(reason) => return exit::parse_error(reason),
    };
    exit::print_json_result(hf_causal_lm_cuda_shared_fork_batch_compare_json(
        path,
        request_count,
        max_context,
        max_new_tokens,
        target_block_tokens,
        min_block_tokens,
        prompt_spec,
        compute_capability,
        experimental_rt,
    ))
}

pub(crate) fn strip_experimental_rt_arg(
    args: &mut impl Iterator<Item = String>,
) -> (Vec<String>, bool) {
    let mut experimental_rt = false;
    let mut items = Vec::new();
    for arg in args {
        if arg == "--experimental-rt" {
            experimental_rt = true;
        } else {
            items.push(arg);
        }
    }
    (items, experimental_rt)
}

pub(crate) fn hf_causal_lm_cuda_shared_fork_batch_json(
    path: Option<String>,
    request_count: usize,
    max_context_tokens: usize,
    max_new_tokens: usize,
    target_block_tokens: usize,
    min_block_tokens: usize,
    prompt_spec: Option<String>,
    compute_capability: Option<u32>,
    experimental_rt: bool,
) -> Result<String, String> {
    let path =
        path.ok_or_else(|| "hf-cuda-shared-fork-batch requires checkpoint_dir".to_string())?;
    let prompt_spec = prompt_spec.unwrap_or_else(|| SHARED_FORK_STORY_PROMPT.to_string());
    let prompt = resolve_prompt_text(&prompt_spec)?;
    let encoded = encode_text_prompt(&path, &prompt)
        .map_err(|err| format!("HF CUDA shared fork batch prompt encode failed: {err}"))?;
    let token_ids = encoded
        .token_ids
        .iter()
        .copied()
        .map(TokenId)
        .collect::<Vec<_>>();
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let output = run_hf_causal_lm_cuda_shared_fork_batch_probe(
        &runtime,
        &path,
        &token_ids,
        request_count,
        max_context_tokens,
        max_new_tokens,
        target_block_tokens,
        min_block_tokens,
        compute_capability,
        true,
        false,
    )
    .map_err(|err| format!("HF CUDA shared fork batch failed: {err:?}"))?;
    shared_fork_batch_json(
        &path,
        &prompt,
        encoded.input_mode,
        &encoded.token_ids,
        &output,
        experimental_rt,
    )
}

pub(crate) fn hf_causal_lm_cuda_shared_fork_batch_compare_json(
    path: Option<String>,
    request_count: usize,
    max_context_tokens: usize,
    max_new_tokens: usize,
    target_block_tokens: usize,
    min_block_tokens: usize,
    prompt_spec: Option<String>,
    compute_capability: Option<u32>,
    experimental_rt: bool,
) -> Result<String, String> {
    let path = path
        .ok_or_else(|| "hf-cuda-shared-fork-batch-compare requires checkpoint_dir".to_string())?;
    let prompt_spec = prompt_spec.unwrap_or_else(|| SHARED_FORK_STORY_PROMPT.to_string());
    let prompt = resolve_prompt_text(&prompt_spec)?;
    let encoded = encode_text_prompt(&path, &prompt)
        .map_err(|err| format!("HF CUDA shared fork batch compare prompt encode failed: {err}"))?;
    let token_ids = encoded
        .token_ids
        .iter()
        .copied()
        .map(TokenId)
        .collect::<Vec<_>>();
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let sequential_single = run_hf_causal_lm_cuda_shared_fork_batch_probe(
        &runtime,
        &path,
        &token_ids,
        1,
        max_context_tokens,
        max_new_tokens,
        1,
        2,
        compute_capability,
        false,
        false,
    )
    .map_err(|err| format!("HF CUDA shared fork sequential baseline failed: {err:?}"))?;
    let sequential = expanded_single_request_baseline(sequential_single, request_count);
    let batched = run_hf_causal_lm_cuda_shared_fork_batch_probe(
        &runtime,
        &path,
        &token_ids,
        request_count,
        max_context_tokens,
        max_new_tokens,
        target_block_tokens,
        min_block_tokens,
        compute_capability,
        true,
        false,
    )
    .map_err(|err| format!("HF CUDA shared fork batched run failed: {err:?}"))?;
    shared_fork_batch_compare_json(
        &path,
        &prompt,
        encoded.input_mode,
        &encoded.token_ids,
        &sequential,
        &batched,
        experimental_rt,
    )
}

fn resolve_prompt_text(prompt_spec: &str) -> Result<String, String> {
    let Some(path) = prompt_spec.strip_prefix('@') else {
        return Ok(prompt_spec.to_string());
    };
    if path.is_empty() {
        return Err("hf-cuda-shared-fork-batch prompt file path is empty".to_string());
    }
    fs::read_to_string(path).map_err(|err| {
        format!("hf-cuda-shared-fork-batch failed to read prompt file {path}: {err}")
    })
}

fn expanded_single_request_baseline(
    mut output: HfCudaSharedForkBatchOutput,
    request_count: usize,
) -> HfCudaSharedForkBatchOutput {
    if request_count <= 1 {
        return output;
    }
    let tokens = output
        .tokens_by_request
        .first()
        .cloned()
        .unwrap_or_default();
    let stopped = output.stopped_by_request.first().copied().unwrap_or(false);
    output.request_count = request_count;
    output.tokens_by_request = vec![tokens; request_count];
    output.stopped_by_request = vec![stopped; request_count];
    output.first_decode_wall_ns = output
        .first_decode_wall_ns
        .saturating_mul(request_count as u64);
    output.continuous_decode_wall_ns = output
        .continuous_decode_wall_ns
        .saturating_mul(request_count as u64);
    output
}

fn shared_fork_batch_compare_json(
    path: &str,
    prompt: &str,
    input_mode: &str,
    prompt_ids: &[u32],
    sequential: &HfCudaSharedForkBatchOutput,
    batched: &HfCudaSharedForkBatchOutput,
    experimental_rt: bool,
) -> Result<String, String> {
    let speedup = ratio(batched.tokens_per_second(), sequential.tokens_per_second());
    let decode_wall_speedup = ratio(
        sequential.decode_wall_ns() as f64,
        batched.decode_wall_ns() as f64,
    );
    let experimental_rt_json = experimental_rt_json(
        experimental_rt,
        batched.request_count,
        batched.max_new_tokens,
        batched.target_block_tokens,
        batched.min_block_tokens,
        prompt_ids.len(),
    );
    Ok(format!(
        "{{\"status\":\"ok\",\"backend\":\"cuda\",\"mode\":\"shared_fork_batch_compare\",\"path\":\"{}\",\"input_mode\":\"{}\",\"prompt\":\"{}\",\"prompt_token_ids\":{},\"prompt_tokens\":{},\"request_count\":{},\"max_new_tokens\":{},\"target_block_tokens\":{},\"min_block_tokens\":{},\"token_match\":{},\"throughput_speedup\":{:.6},\"decode_wall_speedup\":{:.6},\"experimental_rt\":{},\"sequential\":{},\"batched\":{}}}",
        json_escape(path),
        input_mode,
        json_escape(prompt),
        u32s_json(prompt_ids),
        prompt_ids.len(),
        batched.request_count,
        batched.max_new_tokens,
        batched.target_block_tokens,
        batched.min_block_tokens,
        sequential.tokens_by_request == batched.tokens_by_request,
        speedup,
        decode_wall_speedup,
        experimental_rt_json,
        shared_fork_batch_run_json("sequential", sequential)?,
        shared_fork_batch_run_json("batched", batched)?,
    ))
}

fn shared_fork_batch_run_json(
    label: &str,
    output: &HfCudaSharedForkBatchOutput,
) -> Result<String, String> {
    let dtype = nerva_model::precision::bits::dtype_label(output.dtype)
        .map_err(|err| format!("HF CUDA shared fork batch dtype failed: {err:?}"))?;
    Ok(format!(
        "{{\"label\":\"{}\",\"total_tokens\":{},\"tokens_per_second\":{:.6},\"decode_wall_ns\":{},\"load_wall_ns\":{},\"prefill_wall_ns\":{},\"used_batched_projection\":{},\"dtype\":\"{}\",\"layers\":{},\"hidden\":{},\"vocab_size\":{},\"scheduler\":{},\"tokens_by_request\":{},\"stopped_by_request\":{}}}",
        label,
        output.total_tokens(),
        output.tokens_per_second(),
        output.decode_wall_ns(),
        output.load_wall_ns,
        output.prefill_wall_ns,
        output.used_batched_projection(),
        dtype,
        output.metadata.num_hidden_layers,
        output.metadata.hidden_size,
        output.metadata.vocab_size,
        scheduler_json(&output.scheduler),
        tokens_by_request_json(output),
        bools_json(&output.stopped_by_request),
    ))
}

fn shared_fork_batch_json(
    path: &str,
    prompt: &str,
    input_mode: &str,
    prompt_ids: &[u32],
    output: &HfCudaSharedForkBatchOutput,
    experimental_rt: bool,
) -> Result<String, String> {
    let dtype = nerva_model::precision::bits::dtype_label(output.dtype)
        .map_err(|err| format!("HF CUDA shared fork batch dtype failed: {err:?}"))?;
    let experimental_rt_json = experimental_rt_json(
        experimental_rt,
        output.request_count,
        output.max_new_tokens,
        output.target_block_tokens,
        output.min_block_tokens,
        prompt_ids.len(),
    );
    Ok(format!(
        "{{\"status\":\"ok\",\"backend\":\"cuda\",\"mode\":\"shared_fork_batch\",\"path\":\"{}\",\"input_mode\":\"{}\",\"prompt\":\"{}\",\"prompt_token_ids\":{},\"prompt_tokens\":{},\"request_count\":{},\"max_new_tokens\":{},\"target_block_tokens\":{},\"min_block_tokens\":{},\"total_tokens\":{},\"tokens_per_second\":{:.6},\"used_batched_projection\":{},\"experimental_rt\":{},\"dtype\":\"{}\",\"layers\":{},\"hidden\":{},\"vocab_size\":{},\"manifest_entries\":{},\"shard_plan_entries\":{},\"tensors_loaded\":{},\"bytes_loaded\":{},\"data_hash\":{},\"data_hash_available\":{},\"load_wall_ns\":{},\"prefill_wall_ns\":{},\"first_decode_wall_ns\":{},\"continuous_decode_wall_ns\":{},\"decode_wall_ns\":{},\"scheduler\":{},\"tokens_by_request\":{},\"stopped_by_request\":{},\"resident_weights\":{},\"create\":{},\"fork_creates\":[{}]}}",
        json_escape(path),
        input_mode,
        json_escape(prompt),
        u32s_json(prompt_ids),
        prompt_ids.len(),
        output.request_count,
        output.max_new_tokens,
        output.target_block_tokens,
        output.min_block_tokens,
        output.total_tokens(),
        output.tokens_per_second(),
        output.used_batched_projection(),
        experimental_rt_json,
        dtype,
        output.metadata.num_hidden_layers,
        output.metadata.hidden_size,
        output.metadata.vocab_size,
        output.manifest_entries,
        output.shard_plan_entries,
        output.tensors_loaded,
        output.bytes_loaded,
        output.data_hash,
        output.data_hash_available,
        output.load_wall_ns,
        output.prefill_wall_ns,
        output.first_decode_wall_ns,
        output.continuous_decode_wall_ns,
        output.decode_wall_ns(),
        scheduler_json(&output.scheduler),
        tokens_by_request_json(output),
        bools_json(&output.stopped_by_request),
        output.resident_weights.to_json(),
        output.create.to_json(),
        output
            .fork_creates
            .iter()
            .map(|summary| summary.to_json())
            .collect::<Vec<_>>()
            .join(","),
    ))
}

fn experimental_rt_json(
    enabled: bool,
    request_count: usize,
    max_new_tokens: usize,
    target_block_tokens: usize,
    min_block_tokens: usize,
    prompt_tokens: usize,
) -> String {
    if !enabled {
        return "null".to_string();
    }
    let page_tokens = 16u32;
    let dims = 16u32;
    let context_tokens = prompt_tokens
        .saturating_add(max_new_tokens)
        .max(page_tokens as usize);
    let batched_context_tokens = context_tokens.max(prompt_tokens);
    let big_context_tokens = batched_context_tokens.max(128 * 1024);
    let query_count = request_count.max(1);
    let cases = [
        experimental_rt_case(
            "sequential",
            context_tokens,
            1,
            page_tokens,
            dims,
            target_block_tokens,
            min_block_tokens,
        ),
        experimental_rt_case(
            "batched",
            batched_context_tokens,
            query_count,
            page_tokens,
            dims,
            target_block_tokens,
            min_block_tokens,
        ),
        experimental_rt_case(
            "big_context",
            big_context_tokens,
            query_count,
            page_tokens,
            dims,
            target_block_tokens,
            min_block_tokens,
        ),
    ];
    let real_rt_available = cases
        .iter()
        .any(|(_, summary)| summary.real_rt_backend_available);
    let rt_core_capable = cases.iter().any(|(_, summary)| summary.rt_core_capable);
    let case_json = cases
        .iter()
        .map(|(label, summary)| {
            format!(
                "{{\"label\":\"{}\",\"summary\":{}}}",
                json_escape(label),
                summary.to_json()
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"requested\":true,\"scope\":\"attention_stage_synthetic\",\"note\":\"RT cores are used for page candidate selection; summary includes exact CUDA rerank, local/far sparse attention, and online-softmax merge timing.\",\"real_rt_backend_available\":{},\"rt_core_capable\":{},\"page_tokens\":{},\"dims\":{},\"cases\":[{}]}}",
        real_rt_available, rt_core_capable, page_tokens, dims, case_json
    )
}

fn experimental_rt_case(
    label: &'static str,
    context_tokens: usize,
    query_count: usize,
    page_tokens: u32,
    dims: u32,
    target_block_tokens: usize,
    min_block_tokens: usize,
) -> (&'static str, CudaExperimentalRtCandidateBenchSummary) {
    let pages = pages_for_context(context_tokens, page_tokens);
    let requested_candidates = target_block_tokens
        .max(min_block_tokens)
        .max(128)
        .min(u32::MAX as usize) as u32;
    let candidates = pages.min(requested_candidates).max(1);
    let iterations = if pages >= 16 * 1024 { 64 } else { 128 };
    let summary = experimental_rt_candidate_bench(
        pages,
        page_tokens,
        dims,
        query_count.min(u32::MAX as usize) as u32,
        candidates,
        iterations,
        8,
    );
    (label, summary)
}

fn pages_for_context(context_tokens: usize, page_tokens: u32) -> u32 {
    let page_tokens = page_tokens.max(1) as usize;
    context_tokens
        .saturating_add(page_tokens - 1)
        .saturating_div(page_tokens)
        .max(1)
        .min(u32::MAX as usize) as u32
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator <= 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn scheduler_json(summary: &HfCudaSharedForkBatchSchedulerSummary) -> String {
    format!(
        "{{\"scheduler_steps\":{},\"batched_steps\":{},\"batch_groups\":{},\"fallback_steps\":{},\"batch_failed_steps\":{},\"observed_tokens\":{},\"batch_observed_tokens\":{},\"fallback_observed_tokens\":{},\"batch_projection_elapsed_ns\":{},\"batch_qkv_elapsed_ns\":{},\"batch_attention_output_elapsed_ns\":{},\"batch_gate_up_elapsed_ns\":{},\"batch_down_elapsed_ns\":{},\"batch_lm_head_elapsed_ns\":{},\"batch_pack_kernel_launches\":{},\"batch_projection_kernel_launches\":{},\"batch_scatter_kernel_launches\":{},\"batch_dependency_kernel_launches\":{},\"batch_experimental_rt_selector_launches\":{},\"batch_sampling_kernel_launches\":{},\"batch_sync_calls\":{},\"batch_hot_path_allocations\":{},\"last_plan_reason\":\"{}\",\"last_batch_reason\":\"{}\"}}",
        summary.scheduler_steps,
        summary.batched_steps,
        summary.batch_groups,
        summary.fallback_steps,
        summary.batch_failed_steps,
        summary.observed_tokens,
        summary.batch_observed_tokens,
        summary.fallback_observed_tokens,
        summary.batch_projection_elapsed_ns,
        summary.batch_qkv_elapsed_ns,
        summary.batch_attention_output_elapsed_ns,
        summary.batch_gate_up_elapsed_ns,
        summary.batch_down_elapsed_ns,
        summary.batch_lm_head_elapsed_ns,
        summary.batch_pack_kernel_launches,
        summary.batch_projection_kernel_launches,
        summary.batch_scatter_kernel_launches,
        summary.batch_dependency_kernel_launches,
        summary.batch_experimental_rt_selector_launches,
        summary.batch_sampling_kernel_launches,
        summary.batch_sync_calls,
        summary.batch_hot_path_allocations,
        summary.last_plan_reason,
        summary.last_batch_reason,
    )
}

fn tokens_by_request_json(output: &HfCudaSharedForkBatchOutput) -> String {
    let rows = output
        .tokens_by_request
        .iter()
        .map(|tokens| {
            let values = tokens.iter().map(|token| token.0).collect::<Vec<_>>();
            u32s_json(&values)
        })
        .collect::<Vec<_>>();
    format!("[{}]", rows.join(","))
}

fn bools_json(values: &[bool]) -> String {
    let items = values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
    format!("[{}]", items.join(","))
}
