use nerva_core::types::id::token::TokenId;
use nerva_model::hf::tokenizer::{
    decode_generated_text, encode_text_prompt, format_prompt_for_model, stop_token_ids,
};
use nerva_runtime::engine::hf_cuda_decode::file_backed::generate::{
    HfCudaDeviceGenerateOutput, HfCudaRtDecodeConfig, HfCudaSamplerConfig,
    run_hf_causal_lm_cuda_shard_backed_device_generate_with_sampler_profiling_rt_and_progress,
};
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

use crate::cli::args::{
    AUTO_CONTEXT_MARGIN, DEFAULT_OUTPUT_TOKENS, DEFAULT_QUEUE_CAPACITY, parse_args,
};
use crate::cli::model::{detect_cuda_compute_capability, resolve_model_path, resolve_prompt_text};
use crate::cli::ui::logger::NervaCliLogger;
use crate::json::json_escape;

pub(crate) struct GenerateResult {
    pub(crate) output: String,
    pub(crate) print_stdout: bool,
}

pub(crate) fn run_generate(args: &[String]) -> Result<GenerateResult, String> {
    let parsed = parse_args(args)?;
    let model = parsed
        .model
        .as_deref()
        .ok_or_else(|| "missing -m/--model".to_string())?;
    let prompt = parsed
        .prompt
        .as_deref()
        .ok_or_else(|| "missing -p/--prompt".to_string())?;
    let model_path = resolve_model_path(model)?;
    let model_path_string = model_path
        .to_str()
        .ok_or_else(|| "model path is not valid UTF-8".to_string())?
        .to_string();
    let raw_prompt = resolve_prompt_text(prompt)?;
    let formatted = format_prompt_for_model(&model_path_string, &raw_prompt, parsed.prompt_format)?;
    let encoded = encode_text_prompt(&model_path_string, &formatted.text)?;
    let prompt_tokens = encoded
        .token_ids
        .iter()
        .copied()
        .map(TokenId)
        .collect::<Vec<_>>();
    let output_tokens = parsed.output_tokens.unwrap_or(DEFAULT_OUTPUT_TOKENS);
    if output_tokens == 0 {
        return Err("-o/--output must be non-zero".to_string());
    }
    let context_tokens = parsed
        .context_tokens
        .unwrap_or_else(|| prompt_tokens.len() + output_tokens + AUTO_CONTEXT_MARGIN);
    if context_tokens < prompt_tokens.len() + output_tokens {
        return Err(format!(
            "-c/--context is too small: need at least {} tokens for prompt({}) + output({})",
            prompt_tokens.len() + output_tokens,
            prompt_tokens.len(),
            output_tokens
        ));
    }
    let compute_capability = parsed
        .compute_capability
        .or_else(detect_cuda_compute_capability);
    if compute_capability.is_none() {
        return Err(format!(
            "CUDA compute capability is unavailable; run with a CUDA-visible GPU. CUDA probe: {}",
            nerva_runtime::capabilities::discovery::cuda_smoke().to_json()
        ));
    }
    let sampler = HfCudaSamplerConfig {
        temperature: parsed.temperature,
        top_p: parsed.top_p,
        top_k: parsed.top_k,
        seed: parsed.seed,
    };
    let mut rt_decode = HfCudaRtDecodeConfig {
        enabled: parsed.rt,
        mode: rt_mode_code(&parsed.rt_mode)?,
        ..HfCudaRtDecodeConfig::default()
    };
    if let Some(page_tokens) = optional_u32_count("--rt-page-tokens", parsed.rt_page_tokens)? {
        rt_decode.page_tokens = page_tokens;
    }
    if let Some(local_window_tokens) =
        optional_u32_count("--rt-local-window", parsed.rt_local_window_tokens)?
    {
        rt_decode.local_window_tokens = local_window_tokens;
    }
    if let Some(sink_tokens) = optional_u32_count("--rt-sink-tokens", parsed.rt_sink_tokens)? {
        rt_decode.sink_tokens = sink_tokens;
    }
    if let Some(far_pages) = optional_u32_count("--rt-far-pages", parsed.rt_far_pages)? {
        let local_pages = ceil_div_u32(rt_decode.local_window_tokens, rt_decode.page_tokens);
        let sink_pages = ceil_div_u32(rt_decode.sink_tokens, rt_decode.page_tokens);
        rt_decode.pages = far_pages
            .saturating_add(local_pages)
            .saturating_add(sink_pages);
    } else if let Some(pages) = optional_u32_count("--rt-pages", parsed.rt_pages)? {
        rt_decode.pages = pages;
    }
    let queue_capacity = parsed.queue_capacity.unwrap_or(DEFAULT_QUEUE_CAPACITY);
    let stop_token_ids = stop_token_ids(&model_path_string)?;
    let mut logger = NervaCliLogger::new(parsed.json, parsed.debug);
    let tui_active = logger.is_tui_active();
    logger.banner();
    logger.configure(
        &model_path,
        formatted.mode,
        prompt_tokens.len(),
        context_tokens,
        output_tokens,
        queue_capacity,
        compute_capability,
        stop_token_ids.len(),
    );
    logger.runtime_init();
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    logger.load_start();
    let start = std::time::Instant::now();
    let _native_progress = logger.native_load_progress_guard();
    let _ticker = logger.ticker_guard();
    let output =
        run_hf_causal_lm_cuda_shard_backed_device_generate_with_sampler_profiling_rt_and_progress(
            &runtime,
            &model_path_string,
            &prompt_tokens,
            context_tokens,
            output_tokens,
            queue_capacity,
            compute_capability,
            sampler,
            parsed.profiling,
            rt_decode,
            |progress| logger.decode_progress(progress),
        )
        .map_err(|err| format!("generation failed: {err:?}"))?;
    if parsed.json {
        logger.finish(&output, start.elapsed());
        return generate_json_output(
            &model_path_string,
            &raw_prompt,
            encoded.input_mode,
            formatted.mode,
            &encoded.token_ids,
            &output,
            sampler,
            start.elapsed(),
        )
        .map(|output| GenerateResult {
            output,
            print_stdout: true,
        });
    }
    let generated_text = decode_generated_text(&model_path_string, output.tokens())?
        .ok_or_else(|| "model generated tokens but tokenizer decode is unavailable".to_string())?;
    logger.finish(&output, start.elapsed());
    logger.generated_text(generated_text.clone());
    Ok(GenerateResult {
        output: generated_text,
        print_stdout: !tui_active,
    })
}

fn generate_json_output(
    path: &str,
    prompt: &str,
    input_mode: &str,
    prompt_mode: &str,
    prompt_ids: &[u32],
    output: &HfCudaDeviceGenerateOutput,
    sampler: HfCudaSamplerConfig,
    elapsed: std::time::Duration,
) -> Result<String, String> {
    let generated_text = decode_generated_text(path, output.tokens())?
        .map(|text| format!("\"{}\"", json_escape(&text)))
        .unwrap_or_else(|| "null".to_string());
    let tokens = output
        .tokens()
        .iter()
        .map(|token| token.0.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let generated_tokens = output.tokens().len();
    let elapsed_wall_ns = elapsed.as_nanos();
    let post_load_wall_ns =
        output.stream.prefill_wall_ns as u128 + output.stream.decode_wall_ns as u128;
    let critical_paths = output
        .stream
        .chunks
        .iter()
        .flat_map(|chunk| chunk.critical_paths.iter())
        .collect::<Vec<_>>();
    let critical_path_wall_ns = critical_paths
        .iter()
        .map(|path| path.wall_latency_ns)
        .sum::<u64>();
    let critical_path_device_ns = critical_paths
        .iter()
        .map(|path| path.device_timeline_active_ns)
        .sum::<u64>();
    let token_critical_paths = critical_paths
        .iter()
        .map(|path| path.to_json())
        .collect::<Vec<_>>()
        .join(",");
    let chunks = output
        .stream
        .chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| {
            let chunk_wall_ns = chunk
                .critical_paths
                .iter()
                .map(|path| path.wall_latency_ns)
                .sum::<u64>();
            let chunk_device_ns = chunk
                .critical_paths
                .iter()
                .map(|path| path.device_timeline_active_ns)
                .sum::<u64>();
            format!(
                "{{\"chunk_index\":{},\"requested_tokens\":{},\"observed_tokens\":{},\"wall_ns\":{},\"device_ns\":{},\"tokens_per_second\":{},\"projection_ns\":{},\"qkv_projection_ns\":{},\"attention_output_projection_ns\":{},\"gate_up_projection_ns\":{},\"down_projection_ns\":{},\"lm_head_projection_ns\":{},\"attention_ns\":{},\"mlp_ns\":{},\"norm_ns\":{},\"sampling_ns\":{},\"graph_nodes\":{},\"graph_replays\":{},\"graph_cache_hits\":{},\"kernel_launches\":{},\"experimental_rt_selector_launches\":{},\"experimental_rt_sparse_attention_chunks\":{},\"experimental_rt_dense_attention_chunks\":{},\"experimental_rt_attention_chunks\":{},\"h2d_bytes\":{},\"d2h_bytes\":{},\"sync_calls\":{},\"host_causality_edges\":{},\"hot_path_allocations\":{}}}",
                index,
                chunk.steps_requested,
                chunk.tokens.len(),
                chunk_wall_ns,
                chunk_device_ns,
                tokens_per_second(chunk.tokens.len(), chunk_wall_ns as u128),
                chunk.projection_ns,
                chunk.qkv_projection_ns,
                chunk.attention_output_projection_ns,
                chunk.gate_up_projection_ns,
                chunk.down_projection_ns,
                chunk.lm_head_projection_ns,
                chunk.attention_ns,
                chunk.mlp_ns,
                chunk.norm_ns,
                chunk.sampling_ns,
                chunk.graph_nodes,
                chunk.graph_replays,
                chunk.graph_cache_hits,
                chunk.kernel_launches,
                chunk.experimental_rt_selector_launches,
                chunk.experimental_rt_sparse_attention_chunks,
                chunk.experimental_rt_dense_attention_chunks,
                chunk.experimental_rt_attention_chunks,
                chunk.h2d_bytes,
                chunk.d2h_bytes,
                chunk.sync_calls,
                chunk.host_causality_edges,
                chunk.hot_path_allocations,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let rt_page_tokens = output.stream.create.experimental_rt_page_tokens;
    let rt_total_pages = output.stream.create.experimental_rt_pages;
    let rt_local_pages = ceil_div_u32(
        output.stream.create.experimental_rt_local_window_tokens,
        rt_page_tokens,
    );
    let rt_sink_pages = ceil_div_u32(
        output.stream.create.experimental_rt_sink_tokens,
        rt_page_tokens,
    );
    let rt_reserved_pages = rt_local_pages.saturating_add(rt_sink_pages);
    let rt_far_pages = rt_total_pages.saturating_sub(rt_reserved_pages);
    let rt_selected_tokens = rt_total_pages.saturating_mul(rt_page_tokens);
    let rt_local_tokens = rt_local_pages.saturating_mul(rt_page_tokens);
    let rt_sink_page_tokens = rt_sink_pages.saturating_mul(rt_page_tokens);
    let rt_far_tokens = rt_far_pages.saturating_mul(rt_page_tokens);
    let rt_selector_policy = rt_selector_policy(
        output.stream.create.experimental_rt_decode_requested,
        output.stream.create.experimental_rt_decode_enabled,
        output.stream.create.experimental_rt_mode,
        experimental_rt_qk_selector_from_env(),
        experimental_rt_qk_fused_selector_from_env(),
    );
    let experimental_rt_qk_selector = experimental_rt_qk_selector_from_env();
    let experimental_rt_qk_fused_selector = experimental_rt_qk_fused_selector_from_env();
    let experimental_prefill_local_window_tokens =
        experimental_prefill_local_window_tokens_from_env();
    Ok(format!(
        "{{\"status\":\"ok\",\"backend\":\"{}\",\"mode\":\"generate\",\"nerva_version\":\"{}\",\"path\":\"{}\",\"input_mode\":\"{}\",\"prompt_mode\":\"{}\",\"sampler\":{{\"temperature\":{},\"top_p\":{},\"top_k\":{},\"seed\":{}}},\"experimental_rt_decode\":{{\"requested\":{},\"enabled\":{},\"mode\":\"{}\",\"selector_policy\":\"{}\",\"query_key_aware_selector\":{},\"query_key_fused_selector\":{},\"page_tokens\":{},\"pages\":{},\"selected_pages\":{},\"local_pages\":{},\"sink_pages\":{},\"far_pages\":{},\"selected_tokens\":{},\"local_page_tokens\":{},\"sink_page_tokens\":{},\"far_tokens\":{},\"local_window_tokens\":{},\"sink_tokens\":{}}},\"prefill_chunk_tokens\":{},\"experimental_prefill_local_window_tokens\":{},\"head_threads\":{},\"prompt\":\"{}\",\"prompt_token_ids\":[{}],\"prompt_tokens\":{},\"max_new_tokens\":{},\"generated_tokens\":{},\"elapsed_wall_ns\":{},\"load_wall_ns\":{},\"prefill_wall_ns\":{},\"prefill_device_elapsed_ns\":{},\"prefill_projection_ns\":{},\"prefill_qkv_projection_ns\":{},\"prefill_attention_output_projection_ns\":{},\"prefill_gate_up_projection_ns\":{},\"prefill_down_projection_ns\":{},\"prefill_lm_head_projection_ns\":{},\"prefill_attention_ns\":{},\"prefill_mlp_ns\":{},\"prefill_norm_ns\":{},\"prefill_sampling_ns\":{},\"decode_wall_ns\":{},\"post_load_wall_ns\":{},\"end_to_end_tokens_per_second\":{},\"post_load_tokens_per_second\":{},\"critical_path_wall_ns\":{},\"critical_path_device_ns\":{},\"critical_path_tokens_per_second\":{},\"tokens\":[{}],\"generated_text\":{},\"stop_reason\":\"{}\",\"hot_path_allocations\":{},\"chunks\":[{}],\"token_critical_paths\":[{}]}}",
        json_escape(output.backend),
        env!("CARGO_PKG_VERSION"),
        json_escape(path),
        input_mode,
        prompt_mode,
        sampler.temperature,
        sampler.top_p,
        sampler.top_k,
        sampler.seed,
        output.stream.create.experimental_rt_decode_requested,
        output.stream.create.experimental_rt_decode_enabled,
        rt_mode_name(output.stream.create.experimental_rt_mode),
        rt_selector_policy,
        experimental_rt_qk_selector,
        experimental_rt_qk_fused_selector,
        rt_page_tokens,
        rt_total_pages,
        rt_total_pages,
        rt_local_pages,
        rt_sink_pages,
        rt_far_pages,
        rt_selected_tokens,
        rt_local_tokens,
        rt_sink_page_tokens,
        rt_far_tokens,
        output.stream.create.experimental_rt_local_window_tokens,
        output.stream.create.experimental_rt_sink_tokens,
        output.stream.create.prefill_chunk_tokens,
        experimental_prefill_local_window_tokens,
        output.stream.create.head_threads,
        json_escape(prompt),
        prompt_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
        prompt_ids.len(),
        output.max_new_tokens,
        generated_tokens,
        elapsed_wall_ns,
        output.stream.load_wall_ns,
        output.stream.prefill_wall_ns,
        output.stream.start.device_elapsed_ns,
        output.stream.start.projection_ns,
        output.stream.start.qkv_projection_ns,
        output.stream.start.attention_output_projection_ns,
        output.stream.start.gate_up_projection_ns,
        output.stream.start.down_projection_ns,
        output.stream.start.lm_head_projection_ns,
        output.stream.start.attention_ns,
        output.stream.start.mlp_ns,
        output.stream.start.norm_ns,
        output.stream.start.sampling_ns,
        output.stream.decode_wall_ns,
        post_load_wall_ns,
        tokens_per_second(generated_tokens, elapsed_wall_ns),
        tokens_per_second(generated_tokens, post_load_wall_ns),
        critical_path_wall_ns,
        critical_path_device_ns,
        tokens_per_second(critical_paths.len(), critical_path_wall_ns as u128),
        tokens,
        generated_text,
        output.stop_reason().as_str(),
        output
            .stream
            .chunks
            .iter()
            .map(|summary| summary.hot_path_allocations)
            .sum::<u64>(),
        chunks,
        token_critical_paths
    ))
}

fn rt_mode_code(mode: &str) -> Result<u32, String> {
    match mode {
        "auto" => Ok(1),
        "shadow" => Ok(2),
        "sparse" => Ok(3),
        _ => Err(format!("invalid --rt-mode: {mode}")),
    }
}

fn optional_u32_count(name: &str, value: Option<usize>) -> Result<Option<u32>, String> {
    value
        .map(|count| u32::try_from(count).map_err(|_| format!("{name} is too large: {count}")))
        .transpose()
}

fn rt_mode_name(mode: u32) -> &'static str {
    match mode {
        2 => "shadow",
        3 => "sparse",
        _ => "auto",
    }
}

fn rt_selector_policy(
    requested: bool,
    enabled: bool,
    mode: u32,
    qk_selector: bool,
    qk_fused_selector: bool,
) -> &'static str {
    if !requested {
        "none"
    } else if !enabled {
        "unavailable_or_dense_fallback"
    } else if qk_selector && qk_fused_selector {
        "cuda_qk_fused_attention_page_selector"
    } else if qk_selector {
        "cuda_qk_representative_page_selector"
    } else if mode == 2 {
        "optix_synthetic_page_pattern_shadow_only"
    } else {
        "optix_synthetic_sink_local_far_page_pattern"
    }
}

fn experimental_rt_qk_selector_from_env() -> bool {
    env_truthy("NERVA_EXPERIMENTAL_RT_QK_SELECTOR")
}

fn experimental_rt_qk_fused_selector_from_env() -> bool {
    experimental_rt_qk_selector_from_env() && env_truthy("NERVA_EXPERIMENTAL_RT_QK_FUSED")
}

fn env_truthy(name: &str) -> bool {
    let Ok(value) = std::env::var(name) else {
        return false;
    };
    matches!(
        value.trim().as_bytes().first().copied(),
        Some(b'1' | b'y' | b'Y' | b't' | b'T')
    )
}

fn experimental_prefill_local_window_tokens_from_env() -> u32 {
    let Ok(value) = std::env::var("NERVA_EXPERIMENTAL_PREFILL_LOCAL_WINDOW_TOKENS") else {
        return 0;
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return 0;
    }
    match trimmed.parse::<u64>() {
        Ok(0) | Err(_) => 0,
        Ok(value) => value.min(u64::from(u32::MAX)) as u32,
    }
}

fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
    if value == 0 || divisor == 0 {
        0
    } else {
        value.saturating_add(divisor - 1) / divisor
    }
}

fn tokens_per_second(tokens: usize, elapsed_ns: u128) -> String {
    if tokens == 0 || elapsed_ns == 0 {
        return "0.0".to_string();
    }
    format!("{:.6}", tokens as f64 * 1_000_000_000.0 / elapsed_ns as f64)
}
