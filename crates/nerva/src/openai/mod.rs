use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use actix_web::{App, HttpRequest, HttpResponse, HttpServer, web};
use nerva_runtime::engine::hf_cuda_decode::file_backed::generate::HfCudaRtDecodeConfig;
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};
use serde_json::{Value, json};
use tokio::sync::Semaphore;

use crate::cli::args::{DEFAULT_OUTPUT_TOKENS, DEFAULT_QUEUE_CAPACITY};
use crate::cli::model::{detect_cuda_compute_capability, resolve_model_path};
use crate::json::json_escape;

mod admin;
mod batches;
mod context_cache;
mod files;
mod generation;
mod mcp;
mod requests;
mod sessions;
mod streaming;
mod types;

pub(crate) use admin::*;
pub(crate) use batches::*;
pub(crate) use context_cache::*;
pub(crate) use files::*;
pub(crate) use generation::*;
pub(crate) use mcp::*;
pub(crate) use requests::*;
pub(crate) use sessions::*;
pub(crate) use streaming::*;
pub(crate) use types::*;

pub(crate) fn run_server(config: ServeConfig) -> Result<(), String> {
    let config = resolve_config(config)?;
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let address = format!("{}:{}", config.host, config.port);
    let workers = config.workers;
    let max_blocking_threads = config.max_blocking_threads;
    let state = web::Data::new(AppState {
        limiter: Arc::new(Semaphore::new(config.max_concurrent_requests)),
        config: Arc::new(config),
        runtime,
        sessions: Mutex::new(HashMap::new()),
        context_cache: Mutex::new(ContextCacheState::default()),
        mcp_servers: Mutex::new(HashMap::new()),
        files: Mutex::new(HashMap::new()),
        batches: Mutex::new(HashMap::new()),
        next_id: AtomicU64::new(1),
        request_count: AtomicU64::new(0),
        generated_tokens: AtomicU64::new(0),
        scheduler_admitted: AtomicU64::new(0),
        scheduler_completed: AtomicU64::new(0),
        scheduler_active: AtomicU64::new(0),
        scheduler_cache_hits: AtomicU64::new(0),
        scheduler_cache_misses: AtomicU64::new(0),
    });
    eprintln!(
        "NERVA OpenAI-compatible API listening on http://{address} model={} concurrency={}",
        state.config.model_id, state.config.max_concurrent_requests
    );
    actix_web::rt::System::new().block_on(async move {
        let mut server = HttpServer::new(move || {
            App::new()
                .app_data(state.clone())
                .app_data(web::JsonConfig::default().limit(8 * 1024 * 1024))
                .route("/health", web::get().to(health))
                .route("/version", web::get().to(version))
                .route("/metrics", web::get().to(metrics))
                .route("/tokenize", web::post().to(tokenize))
                .route("/detokenize", web::post().to(detokenize))
                .route("/v1/tokenize", web::post().to(tokenize))
                .route("/v1/detokenize", web::post().to(detokenize))
                .route("/v1/models", web::get().to(models))
                .route("/v1/models/{model}", web::get().to(model))
                .route("/v1/sessions", web::post().to(create_session))
                .route("/v1/sessions", web::get().to(list_sessions))
                .route("/v1/sessions/{session_id}", web::get().to(get_session))
                .route(
                    "/v1/sessions/{session_id}",
                    web::delete().to(delete_session),
                )
                .route("/v1/context_cache", web::get().to(context_cache_status))
                .route(
                    "/v1/context_cache/{cache_key}",
                    web::delete().to(delete_context_cache),
                )
                .route("/v1/mcp/servers", web::post().to(register_mcp_server))
                .route("/v1/mcp/servers", web::get().to(list_mcp_servers))
                .route(
                    "/v1/mcp/servers/{server_id}",
                    web::delete().to(delete_mcp_server),
                )
                .route(
                    "/v1/mcp/servers/{server_id}/tools",
                    web::get().to(list_mcp_server_tools),
                )
                .route(
                    "/v1/mcp/servers/{server_id}/call",
                    web::post().to(call_mcp_server_tool),
                )
                .route("/v1/mcp/call", web::post().to(call_mcp_tool))
                .route("/v1/completions", web::post().to(completions))
                .route("/v1/chat/completions", web::post().to(chat_completions))
                .route("/v1/responses", web::post().to(responses))
                .route("/v1/embeddings", web::post().to(unsupported_embeddings))
                .route("/pooling", web::post().to(unsupported_pooling))
                .route("/classify", web::post().to(unsupported_pooling))
                .route("/score", web::post().to(unsupported_pooling))
                .route("/v1/score", web::post().to(unsupported_pooling))
                .route("/rerank", web::post().to(unsupported_pooling))
                .route("/v1/rerank", web::post().to(unsupported_pooling))
                .route("/v2/rerank", web::post().to(unsupported_pooling))
                .route(
                    "/v1/audio/transcriptions",
                    web::post().to(unsupported_audio),
                )
                .route("/v1/audio/translations", web::post().to(unsupported_audio))
                .route("/v1/audio/speech", web::post().to(unsupported_audio))
                .route(
                    "/v1/images/generations",
                    web::post().to(unsupported_multimodal),
                )
                .route("/v1/images/edits", web::post().to(unsupported_multimodal))
                .route(
                    "/v1/images/variations",
                    web::post().to(unsupported_multimodal),
                )
                .route("/v1/moderations", web::post().to(unsupported_moderations))
                .route("/v1/batches", web::post().to(create_batch))
                .route("/v1/batches", web::get().to(list_batches))
                .route("/v1/batches/{batch_id}", web::get().to(get_batch))
                .route(
                    "/v1/batches/{batch_id}/cancel",
                    web::post().to(cancel_batch),
                )
                .route("/v1/files", web::post().to(create_file))
                .route("/v1/files", web::get().to(list_files))
                .route("/v1/files/{file_id}", web::get().to(get_file))
                .route("/v1/files/{file_id}", web::delete().to(delete_file))
                .route(
                    "/v1/files/{file_id}/content",
                    web::get().to(get_file_content),
                )
                .route(
                    "/v1/fine_tuning/jobs",
                    web::post().to(unsupported_fine_tuning),
                )
                .route(
                    "/v1/fine_tuning/jobs",
                    web::get().to(unsupported_fine_tuning),
                )
                .route(
                    "/v1/fine_tuning/jobs/{_id}",
                    web::get().to(unsupported_fine_tuning),
                )
                .route(
                    "/v1/fine_tuning/jobs/{_id}/cancel",
                    web::post().to(unsupported_fine_tuning),
                )
                .route("/v1/load_lora_adapter", web::post().to(unsupported_lora))
                .route("/v1/unload_lora_adapter", web::post().to(unsupported_lora))
                .route("/sleep", web::post().to(unsupported_admin_state))
                .route("/wake_up", web::post().to(unsupported_admin_state))
                .route("/is_sleeping", web::get().to(is_sleeping))
                .route("/reset_prefix_cache", web::post().to(reset_context_cache))
                .route("/start_profile", web::post().to(unsupported_admin_state))
                .route("/stop_profile", web::post().to(unsupported_admin_state))
                .default_service(web::to(not_found))
        });
        if let Some(workers) = workers {
            server = server.workers(workers);
        }
        if let Some(max_blocking_threads) = max_blocking_threads {
            server = server.worker_max_blocking_threads(max_blocking_threads);
        }
        server
            .bind(address)
            .map_err(|err| format!("failed to bind HTTP server: {err}"))?
            .run()
            .await
            .map_err(|err| format!("HTTP server failed: {err}"))
    })
}

fn resolve_config(config: ServeConfig) -> Result<ResolvedServeConfig, String> {
    let model_path = resolve_model_path(&config.model)?;
    let model_path = model_path
        .to_str()
        .ok_or_else(|| "model path is not valid UTF-8".to_string())?
        .to_string();
    let mut rt_decode = HfCudaRtDecodeConfig {
        enabled: config.rt,
        mode: rt_mode_code(&config.rt_mode)?,
        ..HfCudaRtDecodeConfig::default()
    };
    if let Some(page_tokens) = optional_u32_count("--rt-page-tokens", config.rt_page_tokens)? {
        rt_decode.page_tokens = page_tokens;
    }
    if let Some(local_window_tokens) =
        optional_u32_count("--rt-local-window", config.rt_local_window_tokens)?
    {
        rt_decode.local_window_tokens = local_window_tokens;
    }
    if let Some(sink_tokens) = optional_u32_count("--rt-sink-tokens", config.rt_sink_tokens)? {
        rt_decode.sink_tokens = sink_tokens;
    }
    if let Some(far_pages) = optional_u32_count("--rt-far-pages", config.rt_far_pages)? {
        let local_pages = ceil_div_u32(rt_decode.local_window_tokens, rt_decode.page_tokens);
        let sink_pages = ceil_div_u32(rt_decode.sink_tokens, rt_decode.page_tokens);
        rt_decode.pages = far_pages
            .saturating_add(local_pages)
            .saturating_add(sink_pages);
    } else if let Some(pages) = optional_u32_count("--rt-pages", config.rt_pages)? {
        rt_decode.pages = pages;
    }
    let compute_capability = config
        .compute_capability
        .or_else(detect_cuda_compute_capability);
    if compute_capability.is_none() {
        return Err(format!(
            "CUDA compute capability is unavailable; run with a CUDA-visible GPU. CUDA probe: {}",
            nerva_runtime::capabilities::discovery::cuda_smoke().to_json()
        ));
    }
    Ok(ResolvedServeConfig {
        model_id: config.model,
        model_path,
        host: config.host,
        port: config.port,
        context_tokens: config.context_tokens,
        default_output_tokens: config.output_tokens.unwrap_or(DEFAULT_OUTPUT_TOKENS),
        queue_capacity: config.queue_capacity.unwrap_or(DEFAULT_QUEUE_CAPACITY),
        compute_capability,
        max_concurrent_requests: config.max_concurrent_requests,
        workers: config.workers,
        max_blocking_threads: config.max_blocking_threads,
        api_key: config.api_key,
        rt_decode,
        profiling: config.profiling,
    })
}

async fn completions(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        require_known_model(&state, &body)?;
        reject_unsupported_generation_options(&body)?;
        let n = request_n(&body)?;
        let prompts = completion_prompts(&body)?;
        let max_tokens = request_max_tokens(&state, &body)?;
        let temperature = request_f32(&body, "temperature", 1.0)?;
        let top_p = request_f32(&body, "top_p", 1.0)?;
        let top_k = request_u32(&body, "top_k", 0)?;
        let seed = request_u64_opt(&body, "seed")?;
        let stop = request_stop_strings(&body)?;
        let session_id = request_optional_string(&body, "session_id")?;
        let cache_key = request_optional_string(&body, "cache_key")?;
        let created = unix_seconds();
        let id = state.next_response_id("cmpl");
        if request_stream(&body) {
            if prompts.len() != 1 || n != 1 {
                return Err(ApiError::unsupported(
                    "streaming completions currently require exactly one prompt and n=1",
                ));
            }
            return generate_text_stream(
                state.clone(),
                GenerateOptions {
                    prompt: prompts.into_iter().next().unwrap_or_else(empty_text_prompt),
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    seed,
                    stop,
                    session_id,
                    cache_key,
                    include_reasoning: false,
                    reasoning_mode: ReasoningMode::None,
                },
                StreamKind::Completion,
                StreamMeta {
                    id: id.clone(),
                    created,
                    model: state.config.model_id.clone(),
                },
            )
            .await;
        }
        let mut choices = Vec::with_capacity(prompts.len().saturating_mul(n));
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        if n > 1
            && prompts.len() == 1
            && shared_fork_batch_supported(temperature, top_p, top_k, seed)
        {
            let generated = generate_text_batch(
                state.clone(),
                GenerateOptions {
                    prompt: prompts.into_iter().next().unwrap_or_else(empty_text_prompt),
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    seed,
                    stop: stop.clone(),
                    session_id: session_id.clone(),
                    cache_key: cache_key.clone(),
                    include_reasoning: false,
                    reasoning_mode: ReasoningMode::None,
                },
                n,
            )
            .await?;
            for item in generated {
                prompt_tokens += item.prompt_tokens;
                completion_tokens += item.token_ids.len();
                choices.push(json!({
                    "text": item.text,
                    "index": choices.len(),
                    "logprobs": null,
                    "finish_reason": item.finish_reason
                }));
            }
        } else {
            for prompt in prompts {
                for _ in 0..n {
                    let index = choices.len();
                    let generated = generate_text(
                        state.clone(),
                        GenerateOptions {
                            prompt: prompt.clone(),
                            max_tokens,
                            temperature,
                            top_p,
                            top_k,
                            seed,
                            stop: stop.clone(),
                            session_id: session_id.clone(),
                            cache_key: cache_key.clone(),
                            include_reasoning: false,
                            reasoning_mode: ReasoningMode::None,
                        },
                    )
                    .await?;
                    prompt_tokens += generated.prompt_tokens;
                    completion_tokens += generated.token_ids.len();
                    choices.push(json!({
                        "text": generated.text,
                        "index": index,
                        "logprobs": null,
                        "finish_reason": generated.finish_reason
                    }));
                }
            }
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": state.config.model_id,
            "choices": choices,
            "usage": usage(prompt_tokens, completion_tokens)
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn chat_completions(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        require_known_model(&state, &body)?;
        reject_unsupported_generation_options_with_tools(&body)?;
        let n = request_n(&body)?;
        let tool_result = execute_request_mcp_tool(state.clone(), &body).await?;
        let prompt = augment_prompt_with_mcp_tool(
            chat_messages_to_prompt(&body)?,
            tool_result.as_ref(),
            &body,
        );
        let created = unix_seconds();
        let id = state.next_response_id("chatcmpl");
        let include_reasoning = request_include_reasoning(&body)?;
        let reasoning_mode = request_reasoning_mode(&state, &body)?;
        let prompt_format = prompt_format_for_reasoning(reasoning_mode);
        let options = GenerateOptions {
            prompt: PromptInput::Text {
                text: prompt,
                format: prompt_format,
            },
            max_tokens: request_max_tokens(&state, &body)?,
            temperature: request_f32(&body, "temperature", 1.0)?,
            top_p: request_f32(&body, "top_p", 1.0)?,
            top_k: request_u32(&body, "top_k", 0)?,
            seed: request_u64_opt(&body, "seed")?,
            stop: request_stop_strings(&body)?,
            session_id: request_optional_string(&body, "session_id")?,
            cache_key: request_optional_string(&body, "cache_key")?,
            include_reasoning,
            reasoning_mode,
        };
        if request_stream(&body) {
            if n != 1 {
                return Err(ApiError::unsupported(
                    "streaming chat completions currently require n=1",
                ));
            }
            return generate_text_stream(
                state.clone(),
                options,
                StreamKind::ChatCompletion,
                StreamMeta {
                    id: id.clone(),
                    created,
                    model: state.config.model_id.clone(),
                },
            )
            .await;
        }
        let response_include_reasoning = options.include_reasoning;
        let response_reasoning_mode = options.reasoning_mode;
        let mut choices = Vec::with_capacity(n);
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        for index in 0..n {
            let generated = generate_text(state.clone(), options.clone()).await?;
            let split = split_generated_reasoning(&generated.text, response_reasoning_mode);
            prompt_tokens += generated.prompt_tokens;
            completion_tokens += generated.token_ids.len();
            let mut message = json!({
                "role": "assistant",
                "content": split.content
            });
            if response_include_reasoning && !split.reasoning.is_empty() {
                message["reasoning"] = json!(split.reasoning);
                message["reasoning_content"] = json!(message["reasoning"].as_str().unwrap_or(""));
            }
            choices.push(json!({
                "index": index,
                "message": message,
                "logprobs": null,
                "finish_reason": generated.finish_reason
            }));
        }
        let mut response = json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": state.config.model_id,
            "choices": choices,
            "usage": usage(prompt_tokens, completion_tokens)
        });
        if let Some(tool_result) = tool_result {
            response["mcp_tool_results"] = json!([mcp_tool_result_json(&tool_result)]);
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(response))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn responses(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        require_known_model(&state, &body)?;
        reject_unsupported_generation_options_with_tools(&body)?;
        let tool_result = execute_request_mcp_tool(state.clone(), &body).await?;
        let prompt = augment_prompt_with_mcp_tool(
            responses_input_to_prompt(&body)?,
            tool_result.as_ref(),
            &body,
        );
        let created = unix_seconds();
        let id = state.next_response_id("resp");
        let include_reasoning = request_include_reasoning(&body)?;
        let reasoning_mode = request_reasoning_mode(&state, &body)?;
        let prompt_format = prompt_format_for_reasoning(reasoning_mode);
        let options = GenerateOptions {
            prompt: PromptInput::Text {
                text: prompt,
                format: prompt_format,
            },
            max_tokens: request_max_tokens(&state, &body)?,
            temperature: request_f32(&body, "temperature", 1.0)?,
            top_p: request_f32(&body, "top_p", 1.0)?,
            top_k: request_u32(&body, "top_k", 0)?,
            seed: request_u64_opt(&body, "seed")?,
            stop: request_stop_strings(&body)?,
            session_id: request_optional_string(&body, "session_id")?,
            cache_key: request_optional_string(&body, "cache_key")?,
            include_reasoning,
            reasoning_mode,
        };
        if request_stream(&body) {
            return generate_text_stream(
                state.clone(),
                options,
                StreamKind::Response,
                StreamMeta {
                    id: id.clone(),
                    created,
                    model: state.config.model_id.clone(),
                },
            )
            .await;
        }
        let response_include_reasoning = options.include_reasoning;
        let response_reasoning_mode = options.reasoning_mode;
        let generated = generate_text(state.clone(), options).await?;
        let split = split_generated_reasoning(&generated.text, response_reasoning_mode);
        let output_id = state.next_response_id("msg");
        let content_id = state.next_response_id("ct");
        let completion_tokens = generated.token_ids.len();
        let mut output = Vec::new();
        if let Some(tool_result) = tool_result.as_ref() {
            output.push(json!({
                "id": state.next_response_id("mcp"),
                "type": "mcp_call",
                "status": "completed",
                "server_id": tool_result.server_id,
                "name": tool_result.name,
                "arguments": tool_result.arguments,
                "output": tool_result.result
            }));
        }
        if response_include_reasoning && !split.reasoning.is_empty() {
            output.push(json!({
                "id": state.next_response_id("rsn"),
                "type": "reasoning",
                "summary": [],
                "status": "completed",
                "content": [{
                    "type": "reasoning_text",
                    "text": split.reasoning
                }]
            }));
        }
        output.push(json!({
            "id": output_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "id": content_id,
                "type": "output_text",
                "text": split.content,
                "annotations": []
            }]
        }));
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "model": state.config.model_id,
            "output": output,
            "output_text": split.content,
            "usage": {
                "input_tokens": generated.prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": generated.prompt_tokens + completion_tokens
            }
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
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

fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
    if value == 0 || divisor == 0 {
        0
    } else {
        value.saturating_add(divisor - 1) / divisor
    }
}

fn stochastic_sampling_requested(temperature: f32, top_k: u32) -> bool {
    temperature > 0.0 && top_k != 1
}

fn auto_sampler_seed() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let pid = std::process::id() as u64;
    let stack_addr = &now as *const _ as usize as u64;
    let mut seed = now.as_nanos() as u64
        ^ now.as_secs().rotate_left(17)
        ^ pid.rotate_left(32)
        ^ stack_addr.rotate_left(7);
    seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    seed = (seed ^ (seed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    seed = (seed ^ (seed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    seed ^ (seed >> 31)
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl AppState {
    fn next_response_id(&self, prefix: &str) -> String {
        let sequence = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("{prefix}-{:x}-{:x}", unix_seconds(), sequence)
    }
}

impl GeneratedText {
    #[allow(dead_code)]
    fn escaped_text(&self) -> String {
        json_escape(&self.text)
    }
}

#[cfg(test)]
mod tests;
