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
    let sampler = HfCudaSamplerConfig {
        temperature: parsed.temperature,
        top_p: parsed.top_p,
        top_k: parsed.top_k,
        seed: parsed.seed,
    };
    let rt_decode = HfCudaRtDecodeConfig {
        enabled: parsed.rt,
        mode: rt_mode_code(&parsed.rt_mode)?,
        ..HfCudaRtDecodeConfig::default()
    };
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
                "{{\"chunk_index\":{},\"requested_tokens\":{},\"observed_tokens\":{},\"wall_ns\":{},\"device_ns\":{},\"tokens_per_second\":{},\"projection_ns\":{},\"qkv_projection_ns\":{},\"attention_output_projection_ns\":{},\"gate_up_projection_ns\":{},\"down_projection_ns\":{},\"lm_head_projection_ns\":{},\"attention_ns\":{},\"mlp_ns\":{},\"norm_ns\":{},\"sampling_ns\":{},\"graph_nodes\":{},\"graph_replays\":{},\"graph_cache_hits\":{},\"kernel_launches\":{},\"h2d_bytes\":{},\"d2h_bytes\":{},\"sync_calls\":{},\"host_causality_edges\":{},\"hot_path_allocations\":{}}}",
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
                chunk.h2d_bytes,
                chunk.d2h_bytes,
                chunk.sync_calls,
                chunk.host_causality_edges,
                chunk.hot_path_allocations,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "{{\"status\":\"ok\",\"backend\":\"cuda\",\"mode\":\"generate\",\"nerva_version\":\"{}\",\"path\":\"{}\",\"input_mode\":\"{}\",\"prompt_mode\":\"{}\",\"sampler\":{{\"temperature\":{},\"top_p\":{},\"top_k\":{},\"seed\":{}}},\"experimental_rt_decode\":{{\"requested\":{},\"enabled\":{},\"page_tokens\":{},\"pages\":{},\"local_window_tokens\":{},\"sink_tokens\":{}}},\"prefill_chunk_tokens\":{},\"head_threads\":{},\"prompt\":\"{}\",\"prompt_token_ids\":[{}],\"prompt_tokens\":{},\"max_new_tokens\":{},\"generated_tokens\":{},\"elapsed_wall_ns\":{},\"load_wall_ns\":{},\"prefill_wall_ns\":{},\"prefill_device_elapsed_ns\":{},\"prefill_projection_ns\":{},\"prefill_qkv_projection_ns\":{},\"prefill_attention_output_projection_ns\":{},\"prefill_gate_up_projection_ns\":{},\"prefill_down_projection_ns\":{},\"prefill_lm_head_projection_ns\":{},\"prefill_attention_ns\":{},\"prefill_mlp_ns\":{},\"prefill_norm_ns\":{},\"prefill_sampling_ns\":{},\"decode_wall_ns\":{},\"post_load_wall_ns\":{},\"end_to_end_tokens_per_second\":{},\"post_load_tokens_per_second\":{},\"critical_path_wall_ns\":{},\"critical_path_device_ns\":{},\"critical_path_tokens_per_second\":{},\"tokens\":[{}],\"generated_text\":{},\"stop_reason\":\"{}\",\"hot_path_allocations\":{},\"chunks\":[{}],\"token_critical_paths\":[{}]}}",
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
        output.stream.create.experimental_rt_page_tokens,
        output.stream.create.experimental_rt_pages,
        output.stream.create.experimental_rt_local_window_tokens,
        output.stream.create.experimental_rt_sink_tokens,
        output.stream.create.prefill_chunk_tokens,
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

fn rt_mode_name(mode: u32) -> &'static str {
    match mode {
        2 => "shadow",
        3 => "sparse",
        _ => "auto",
    }
}

fn tokens_per_second(tokens: usize, elapsed_ns: u128) -> String {
    if tokens == 0 || elapsed_ns == 0 {
        return "0.0".to_string();
    }
    format!("{:.6}", tokens as f64 * 1_000_000_000.0 / elapsed_ns as f64)
}
