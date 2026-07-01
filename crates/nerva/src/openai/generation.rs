use std::sync::atomic::Ordering;

use actix_web::{HttpResponse, web};
use futures_util::StreamExt;
use nerva_core::types::id::token::TokenId;
use nerva_model::hf::tokenizer::{
    decode_generated_text, encode_text_prompt, format_prompt_for_model,
};
use nerva_runtime::engine::hf_cuda_decode::file_backed::generate::{
    HfCudaSamplerConfig,
    run_hf_causal_lm_cuda_shard_backed_device_generate_with_sampler_profiling_rt_and_progress,
};
use nerva_runtime::engine::hf_cuda_decode::file_backed::progress::HfCudaDeviceProgressPhase;
use nerva_runtime::engine::hf_cuda_decode::file_backed::shared_fork_batch::run_hf_causal_lm_cuda_shared_fork_batch_probe;
use nerva_runtime::engine::runtime::Runtime;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::cli::args::{AUTO_CONTEXT_MARGIN, DEFAULT_SEED};

use super::{
    ApiError, AppState, GenerateOptions, GeneratedText, PreparedGeneration, PromptInput,
    ResolvedServeConfig, StreamEmissionState, StreamKind, StreamMeta, StreamRunStats,
    append_response_to_conversation, apply_stop_strings, auto_sampler_seed, completion_text,
    emit_stream_text, ensure_session_for_request, finish_reason, record_context_cache_probe,
    record_session_generation, response_stream_completed_response, send_stream_done,
    send_stream_error, send_stream_final, stochastic_sampling_requested,
    store_response_if_requested, validate_sampling,
};

pub(crate) async fn generate_text_stream(
    state: web::Data<AppState>,
    options: GenerateOptions,
    kind: StreamKind,
    meta: StreamMeta,
) -> Result<HttpResponse, ApiError> {
    state.scheduler_admitted.fetch_add(1, Ordering::Relaxed);
    let permit = state
        .limiter
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::internal("request limiter closed"))?;
    let (tx, rx) = mpsc::channel::<web::Bytes>(32);
    let config = state.config.clone();
    let runtime = state.runtime.clone();
    let state_for_engine = state.clone();
    actix_web::rt::task::spawn_blocking(move || {
        let _permit = permit;
        state_for_engine
            .scheduler_active
            .fetch_add(1, Ordering::Relaxed);
        match generate_text_stream_sync(
            &state_for_engine,
            &runtime,
            &config,
            options,
            kind,
            meta,
            tx.clone(),
        ) {
            Ok(stats) => {
                state_for_engine
                    .generated_tokens
                    .fetch_add(stats.generated_tokens as u64, Ordering::Relaxed);
                record_session_generation(&state_for_engine, &stats);
            }
            Err(error) => {
                send_stream_error(&tx, error);
            }
        }
        state_for_engine
            .scheduler_active
            .fetch_sub(1, Ordering::Relaxed);
        state_for_engine
            .scheduler_completed
            .fetch_add(1, Ordering::Relaxed);
    });
    let body = ReceiverStream::new(rx).map(Ok::<web::Bytes, actix_web::Error>);
    Ok(HttpResponse::Ok()
        .insert_header(("content-type", "text/event-stream"))
        .insert_header(("cache-control", "no-cache"))
        .insert_header(("x-accel-buffering", "no"))
        .streaming(body))
}

fn generate_text_stream_sync(
    state: &AppState,
    runtime: &Runtime,
    config: &ResolvedServeConfig,
    options: GenerateOptions,
    kind: StreamKind,
    meta: StreamMeta,
    tx: mpsc::Sender<web::Bytes>,
) -> Result<StreamRunStats, ApiError> {
    let prepared = prepare_generation(state, config, &options)?;
    let mut streamed_tokens = Vec::new();
    let mut emitted = StreamEmissionState::default();
    let mut stopped_by_stop_string = false;
    let mut tx_closed = false;
    let include_reasoning = options.include_reasoning;
    let reasoning_mode = options.reasoning_mode;
    let output =
        run_hf_causal_lm_cuda_shard_backed_device_generate_with_sampler_profiling_rt_and_progress(
            runtime,
            &config.model_path,
            &prepared.prompt_tokens,
            prepared.context_tokens,
            options.max_tokens,
            config.queue_capacity,
            config.compute_capability,
            prepared.sampler,
            config.profiling,
            config.rt_decode,
            |progress| {
                if tx_closed
                    || stopped_by_stop_string
                    || progress.phase != HfCudaDeviceProgressPhase::Decode
                    || progress.tokens.is_empty()
                {
                    return;
                }
                streamed_tokens.extend(progress.tokens.iter().copied());
                let Ok(Some(current_text)) =
                    decode_generated_text(&config.model_path, &streamed_tokens)
                else {
                    return;
                };
                let (limited_text, hit_stop) = apply_stop_strings(current_text, &options.stop);
                let limited_text =
                    completion_text(&limited_text, options.output_prefix.as_deref(), None);
                if !emit_stream_text(
                    &tx,
                    kind,
                    &meta,
                    include_reasoning,
                    reasoning_mode,
                    &mut emitted,
                    &limited_text,
                ) {
                    tx_closed = true;
                    return;
                }
                stopped_by_stop_string = hit_stop;
            },
        )
        .map_err(|err| ApiError::internal(format!("generation failed: {err:?}")))?;
    if !tx_closed {
        let final_text = decode_generated_text(&config.model_path, output.tokens())
            .map_err(ApiError::bad_request)?
            .ok_or_else(|| ApiError::bad_request("tokenizer decode is unavailable"))?;
        let (limited_text, hit_stop) = apply_stop_strings(final_text, &options.stop);
        let limited_text = completion_text(
            &limited_text,
            options.output_prefix.as_deref(),
            options.output_suffix.as_deref(),
        );
        stopped_by_stop_string |= hit_stop;
        tx_closed = !emit_stream_text(
            &tx,
            kind,
            &meta,
            include_reasoning,
            reasoning_mode,
            &mut emitted,
            &limited_text,
        );
    }
    if !tx_closed {
        let finish_reason = finish_reason(output.stop_reason(), stopped_by_stop_string).to_string();
        let mut completed_response = response_stream_completed_response(
            &meta,
            &emitted,
            prepared.prompt_tokens.len(),
            output.tokens().len(),
        );
        if let (Some(response_options), Some(response_json)) =
            (meta.response.as_ref(), completed_response.take())
        {
            completed_response = Some(store_response_if_requested(
                state,
                response_json,
                response_options.input_items.clone(),
                response_options.store,
            )?);
            if let Some(response_json) = completed_response.as_ref() {
                append_response_to_conversation(
                    state,
                    response_options.conversation_id.as_deref(),
                    &response_options.input_items,
                    response_json,
                )?;
            }
        }
        send_stream_final(
            &tx,
            kind,
            &meta,
            &finish_reason,
            prepared.prompt_tokens.len(),
            output.tokens().len(),
            completed_response,
        );
        send_stream_done(&tx);
    }
    Ok(StreamRunStats {
        generated_tokens: output.tokens().len(),
        prompt_tokens: prepared.prompt_tokens.len(),
        prompt_hash: prepared.prompt_hash,
        cache_key: prepared.cache_key,
        cache_hit: prepared.cache_hit,
        session_id: prepared.session_id,
    })
}

pub(crate) async fn generate_text(
    state: web::Data<AppState>,
    options: GenerateOptions,
) -> Result<GeneratedText, ApiError> {
    state.scheduler_admitted.fetch_add(1, Ordering::Relaxed);
    let permit = state
        .limiter
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::internal("request limiter closed"))?;
    let config = state.config.clone();
    let runtime = state.runtime.clone();
    let state_for_engine = state.clone();
    let result = web::block(move || {
        let _permit = permit;
        state_for_engine
            .scheduler_active
            .fetch_add(1, Ordering::Relaxed);
        let result = generate_text_sync(&state_for_engine, &runtime, &config, options);
        state_for_engine
            .scheduler_active
            .fetch_sub(1, Ordering::Relaxed);
        state_for_engine
            .scheduler_completed
            .fetch_add(1, Ordering::Relaxed);
        result
    })
    .await
    .map_err(|err| ApiError::internal(format!("generation task failed: {err}")))?;
    let generated = result?;
    state
        .generated_tokens
        .fetch_add(generated.token_ids.len() as u64, Ordering::Relaxed);
    record_session_generation(
        &state,
        &StreamRunStats {
            generated_tokens: generated.token_ids.len(),
            prompt_tokens: generated.prompt_tokens,
            prompt_hash: generated.prompt_hash,
            cache_key: generated.cache_key.clone(),
            cache_hit: generated.cache_hit,
            session_id: generated.session_id.clone(),
        },
    );
    Ok(generated)
}

pub(crate) async fn generate_text_batch(
    state: web::Data<AppState>,
    options: GenerateOptions,
    request_count: usize,
) -> Result<Vec<GeneratedText>, ApiError> {
    state.scheduler_admitted.fetch_add(1, Ordering::Relaxed);
    let permit = state
        .limiter
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::internal("request limiter closed"))?;
    let config = state.config.clone();
    let runtime = state.runtime.clone();
    let state_for_engine = state.clone();
    let result = web::block(move || {
        let _permit = permit;
        state_for_engine
            .scheduler_active
            .fetch_add(1, Ordering::Relaxed);
        let result =
            generate_text_batch_sync(&state_for_engine, &runtime, &config, options, request_count);
        state_for_engine
            .scheduler_active
            .fetch_sub(1, Ordering::Relaxed);
        state_for_engine
            .scheduler_completed
            .fetch_add(1, Ordering::Relaxed);
        result
    })
    .await
    .map_err(|err| ApiError::internal(format!("batch generation task failed: {err}")))?;
    let generated = result?;
    let total_generated = generated
        .iter()
        .map(|item| item.token_ids.len())
        .sum::<usize>();
    state
        .generated_tokens
        .fetch_add(total_generated as u64, Ordering::Relaxed);
    for item in &generated {
        record_session_generation(
            &state,
            &StreamRunStats {
                generated_tokens: item.token_ids.len(),
                prompt_tokens: item.prompt_tokens,
                prompt_hash: item.prompt_hash,
                cache_key: item.cache_key.clone(),
                cache_hit: item.cache_hit,
                session_id: item.session_id.clone(),
            },
        );
    }
    Ok(generated)
}

fn generate_text_batch_sync(
    state: &AppState,
    runtime: &Runtime,
    config: &ResolvedServeConfig,
    options: GenerateOptions,
    request_count: usize,
) -> Result<Vec<GeneratedText>, ApiError> {
    if request_count == 0 {
        return Err(ApiError::bad_request(
            "batch request_count must be non-zero",
        ));
    }
    let prepared = prepare_generation(state, config, &options)?;
    let output = run_hf_causal_lm_cuda_shared_fork_batch_probe(
        runtime,
        &config.model_path,
        &prepared.prompt_tokens,
        request_count,
        prepared.context_tokens,
        options.max_tokens,
        32,
        1,
        config.compute_capability,
        true,
        config.profiling,
    )
    .map_err(|err| ApiError::internal(format!("shared-fork batch generation failed: {err:?}")))?;
    output
        .tokens_by_request
        .iter()
        .enumerate()
        .map(|(index, tokens)| {
            let token_ids = tokens.iter().map(|token| token.0).collect::<Vec<_>>();
            let decoded = decode_generated_text(&config.model_path, tokens)
                .map_err(ApiError::bad_request)?
                .ok_or_else(|| ApiError::bad_request("tokenizer decode is unavailable"))?;
            let (text, stopped_by_stop_string) = apply_stop_strings(decoded, &options.stop);
            let text = completion_text(
                &text,
                options.output_prefix.as_deref(),
                options.output_suffix.as_deref(),
            );
            let finish_reason = if stopped_by_stop_string
                || output
                    .stopped_by_request
                    .get(index)
                    .copied()
                    .unwrap_or(false)
                    && tokens.len() < options.max_tokens
            {
                "stop"
            } else {
                "length"
            };
            Ok(GeneratedText {
                text,
                token_ids,
                prompt_tokens: prepared.prompt_tokens.len(),
                finish_reason,
                prompt_hash: prepared.prompt_hash,
                cache_key: prepared.cache_key.clone(),
                cache_hit: prepared.cache_hit,
                session_id: prepared.session_id.clone(),
            })
        })
        .collect()
}

fn generate_text_sync(
    state: &AppState,
    runtime: &Runtime,
    config: &ResolvedServeConfig,
    options: GenerateOptions,
) -> Result<GeneratedText, ApiError> {
    let prepared = prepare_generation(state, config, &options)?;
    let output =
        run_hf_causal_lm_cuda_shard_backed_device_generate_with_sampler_profiling_rt_and_progress(
            runtime,
            &config.model_path,
            &prepared.prompt_tokens,
            prepared.context_tokens,
            options.max_tokens,
            config.queue_capacity,
            config.compute_capability,
            prepared.sampler,
            config.profiling,
            config.rt_decode,
            |_| {},
        )
        .map_err(|err| ApiError::internal(format!("generation failed: {err:?}")))?;
    let token_ids = output
        .tokens()
        .iter()
        .map(|token| token.0)
        .collect::<Vec<_>>();
    let decoded = decode_generated_text(&config.model_path, output.tokens())
        .map_err(ApiError::bad_request)?
        .ok_or_else(|| ApiError::bad_request("tokenizer decode is unavailable"))?;
    let (text, stopped_by_stop_string) = apply_stop_strings(decoded, &options.stop);
    let text = completion_text(
        &text,
        options.output_prefix.as_deref(),
        options.output_suffix.as_deref(),
    );
    let finish_reason = finish_reason(output.stop_reason(), stopped_by_stop_string);
    Ok(GeneratedText {
        text,
        token_ids,
        prompt_tokens: prepared.prompt_tokens.len(),
        finish_reason,
        prompt_hash: prepared.prompt_hash,
        cache_key: prepared.cache_key,
        cache_hit: prepared.cache_hit,
        session_id: prepared.session_id,
    })
}

pub(crate) fn shared_fork_batch_supported(
    temperature: f32,
    top_p: f32,
    top_k: u32,
    seed: Option<u64>,
) -> bool {
    temperature == 0.0 && top_p == 1.0 && top_k == 0 && seed.unwrap_or(DEFAULT_SEED) == DEFAULT_SEED
}

fn prepare_generation(
    state: &AppState,
    config: &ResolvedServeConfig,
    options: &GenerateOptions,
) -> Result<PreparedGeneration, ApiError> {
    if options.max_tokens == 0 {
        return Err(ApiError::bad_request("max_tokens must be non-zero"));
    }
    validate_sampling(options.temperature, options.top_p)?;
    let prompt_tokens = match &options.prompt {
        PromptInput::Text { text, format } => {
            let formatted = format_prompt_for_model(&config.model_path, text, *format)
                .map_err(ApiError::bad_request)?;
            let encoded = encode_text_prompt(&config.model_path, &formatted.text)
                .map_err(ApiError::bad_request)?;
            encoded
                .token_ids
                .iter()
                .copied()
                .map(TokenId)
                .collect::<Vec<_>>()
        }
        PromptInput::TokenIds(tokens) => tokens.clone(),
    };
    if prompt_tokens.is_empty() {
        return Err(ApiError::bad_request("prompt token list must not be empty"));
    }
    let context_tokens = config.context_tokens.unwrap_or_else(|| {
        prompt_tokens
            .len()
            .saturating_add(options.max_tokens)
            .saturating_add(AUTO_CONTEXT_MARGIN)
    });
    if context_tokens < prompt_tokens.len().saturating_add(options.max_tokens) {
        return Err(ApiError::bad_request(format!(
            "context window is too small: need at least {} tokens for prompt({}) + output({})",
            prompt_tokens.len() + options.max_tokens,
            prompt_tokens.len(),
            options.max_tokens
        )));
    }
    let sampler = HfCudaSamplerConfig {
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        seed: options.seed.unwrap_or_else(|| {
            if stochastic_sampling_requested(options.temperature, options.top_k) {
                auto_sampler_seed()
            } else {
                DEFAULT_SEED
            }
        }),
    };
    let prompt_hash = hash_tokens(&prompt_tokens);
    let cache_key = options
        .cache_key
        .clone()
        .unwrap_or_else(|| format!("prompt:{prompt_hash:016x}"));
    let cache_hit =
        record_context_cache_probe(state, &cache_key, prompt_hash, prompt_tokens.len())?;
    let session_id = ensure_session_for_request(state, options.session_id.as_deref())?;
    Ok(PreparedGeneration {
        prompt_tokens,
        context_tokens,
        sampler,
        prompt_hash,
        cache_key,
        cache_hit,
        session_id,
    })
}

pub(crate) fn generate_text_direct_sync(
    state: &AppState,
    options: GenerateOptions,
) -> Result<GeneratedText, ApiError> {
    state.scheduler_admitted.fetch_add(1, Ordering::Relaxed);
    state.scheduler_active.fetch_add(1, Ordering::Relaxed);
    let result = generate_text_sync(state, &state.runtime, state.config.as_ref(), options);
    state.scheduler_active.fetch_sub(1, Ordering::Relaxed);
    state.scheduler_completed.fetch_add(1, Ordering::Relaxed);
    if let Ok(generated) = &result {
        state
            .generated_tokens
            .fetch_add(generated.token_ids.len() as u64, Ordering::Relaxed);
        record_session_generation(
            state,
            &StreamRunStats {
                generated_tokens: generated.token_ids.len(),
                prompt_tokens: generated.prompt_tokens,
                prompt_hash: generated.prompt_hash,
                cache_key: generated.cache_key.clone(),
                cache_hit: generated.cache_hit,
                session_id: generated.session_id.clone(),
            },
        );
    }
    result
}

pub(crate) fn generate_text_batch_direct_sync(
    state: &AppState,
    options: GenerateOptions,
    request_count: usize,
) -> Result<Vec<GeneratedText>, ApiError> {
    state.scheduler_admitted.fetch_add(1, Ordering::Relaxed);
    state.scheduler_active.fetch_add(1, Ordering::Relaxed);
    let result = generate_text_batch_sync(
        state,
        &state.runtime,
        state.config.as_ref(),
        options,
        request_count,
    );
    state.scheduler_active.fetch_sub(1, Ordering::Relaxed);
    state.scheduler_completed.fetch_add(1, Ordering::Relaxed);
    if let Ok(generated) = &result {
        let total_generated = generated
            .iter()
            .map(|item| item.token_ids.len())
            .sum::<usize>();
        state
            .generated_tokens
            .fetch_add(total_generated as u64, Ordering::Relaxed);
        for item in generated {
            record_session_generation(
                state,
                &StreamRunStats {
                    generated_tokens: item.token_ids.len(),
                    prompt_tokens: item.prompt_tokens,
                    prompt_hash: item.prompt_hash,
                    cache_key: item.cache_key.clone(),
                    cache_hit: item.cache_hit,
                    session_id: item.session_id.clone(),
                },
            );
        }
    }
    result
}

pub(crate) fn hash_tokens(tokens: &[TokenId]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for token in tokens {
        hash ^= u64::from(token.0);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
