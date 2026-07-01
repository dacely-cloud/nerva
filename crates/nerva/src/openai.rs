use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use actix_web::http::StatusCode;
use actix_web::{App, HttpRequest, HttpResponse, HttpServer, web};
use nerva_core::types::id::token::TokenId;
use nerva_model::causal_lm::types::HfCausalLmStopReason;
use nerva_model::hf::tokenizer::{
    PromptFormat, decode_generated_text, encode_text_prompt, format_prompt_for_model,
};
use nerva_runtime::engine::hf_cuda_decode::file_backed::generate::{
    HfCudaRtDecodeConfig, HfCudaSamplerConfig,
    run_hf_causal_lm_cuda_shard_backed_device_generate_with_sampler_profiling_rt_and_progress,
};
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};
use serde_json::{Value, json};
use tokio::sync::Semaphore;

use crate::cli::args::{
    AUTO_CONTEXT_MARGIN, DEFAULT_OUTPUT_TOKENS, DEFAULT_QUEUE_CAPACITY, DEFAULT_SEED,
};
use crate::cli::model::{detect_cuda_compute_capability, resolve_model_path};
use crate::json::json_escape;

#[derive(Clone, Debug)]
pub(crate) struct ServeConfig {
    pub model: String,
    pub host: String,
    pub port: u16,
    pub context_tokens: Option<usize>,
    pub output_tokens: Option<usize>,
    pub queue_capacity: Option<usize>,
    pub compute_capability: Option<u32>,
    pub max_concurrent_requests: usize,
    pub workers: Option<usize>,
    pub max_blocking_threads: Option<usize>,
    pub api_key: Option<String>,
    pub rt: bool,
    pub rt_mode: String,
    pub rt_page_tokens: Option<usize>,
    pub rt_pages: Option<usize>,
    pub rt_far_pages: Option<usize>,
    pub rt_local_window_tokens: Option<usize>,
    pub rt_sink_tokens: Option<usize>,
    pub profiling: bool,
}

#[derive(Clone, Debug)]
struct ResolvedServeConfig {
    model_id: String,
    model_path: String,
    host: String,
    port: u16,
    context_tokens: Option<usize>,
    default_output_tokens: usize,
    queue_capacity: usize,
    compute_capability: Option<u32>,
    max_concurrent_requests: usize,
    workers: Option<usize>,
    max_blocking_threads: Option<usize>,
    api_key: Option<String>,
    rt_decode: HfCudaRtDecodeConfig,
    profiling: bool,
}

struct AppState {
    config: Arc<ResolvedServeConfig>,
    runtime: Runtime,
    limiter: Arc<Semaphore>,
    next_id: AtomicU64,
    request_count: AtomicU64,
    generated_tokens: AtomicU64,
}

#[derive(Clone, Debug)]
struct GenerateOptions {
    prompt: String,
    prompt_format: PromptFormat,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    seed: Option<u64>,
    stop: Vec<String>,
}

#[derive(Clone, Debug)]
struct GeneratedText {
    text: String,
    token_ids: Vec<u32>,
    prompt_tokens: usize,
    finish_reason: &'static str,
}

#[derive(Clone, Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_request_error",
            message: message.into(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "not_found_error",
            message: message.into(),
        }
    }

    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "authentication_error",
            message: "invalid or missing bearer token".to_string(),
        }
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            code: "unsupported_operation",
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "server_error",
            message: message.into(),
        }
    }

    fn into_response(self) -> HttpResponse {
        HttpResponse::build(self.status).json(json!({
            "error": {
                "message": self.message,
                "type": self.code,
                "param": null,
                "code": self.code
            }
        }))
    }
}

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
        next_id: AtomicU64::new(1),
        request_count: AtomicU64::new(0),
        generated_tokens: AtomicU64::new(0),
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
                .route("/v1/audio/transcriptions", web::post().to(unsupported_audio))
                .route("/v1/audio/translations", web::post().to(unsupported_audio))
                .route("/v1/audio/speech", web::post().to(unsupported_audio))
                .route("/v1/images/generations", web::post().to(unsupported_multimodal))
                .route("/v1/images/edits", web::post().to(unsupported_multimodal))
                .route("/v1/images/variations", web::post().to(unsupported_multimodal))
                .route("/v1/moderations", web::post().to(unsupported_moderations))
                .route("/v1/batches", web::post().to(unsupported_batch_api))
                .route("/v1/batches", web::get().to(unsupported_batch_api))
                .route("/v1/batches/{_id}", web::get().to(unsupported_batch_api))
                .route("/v1/batches/{_id}/cancel", web::post().to(unsupported_batch_api))
                .route("/v1/files", web::post().to(unsupported_files_api))
                .route("/v1/files", web::get().to(unsupported_files_api))
                .route("/v1/files/{_id}", web::get().to(unsupported_files_api))
                .route("/v1/files/{_id}", web::delete().to(unsupported_files_api))
                .route("/v1/files/{_id}/content", web::get().to(unsupported_files_api))
                .route("/v1/fine_tuning/jobs", web::post().to(unsupported_fine_tuning))
                .route("/v1/fine_tuning/jobs", web::get().to(unsupported_fine_tuning))
                .route("/v1/fine_tuning/jobs/{_id}", web::get().to(unsupported_fine_tuning))
                .route("/v1/fine_tuning/jobs/{_id}/cancel", web::post().to(unsupported_fine_tuning))
                .route("/v1/load_lora_adapter", web::post().to(unsupported_lora))
                .route("/v1/unload_lora_adapter", web::post().to(unsupported_lora))
                .route("/sleep", web::post().to(unsupported_admin_state))
                .route("/wake_up", web::post().to(unsupported_admin_state))
                .route("/is_sleeping", web::get().to(is_sleeping))
                .route("/reset_prefix_cache", web::post().to(unsupported_admin_state))
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

async fn health() -> HttpResponse {
    HttpResponse::Ok().body("OK")
}

async fn version() -> HttpResponse {
    HttpResponse::Ok().json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "engine": "nerva"
    }))
}

async fn metrics(state: web::Data<AppState>) -> HttpResponse {
    let body = format!(
        concat!(
            "# TYPE nerva_openai_requests_total counter\n",
            "nerva_openai_requests_total {}\n",
            "# TYPE nerva_openai_generated_tokens_total counter\n",
            "nerva_openai_generated_tokens_total {}\n"
        ),
        state.request_count.load(Ordering::Relaxed),
        state.generated_tokens.load(Ordering::Relaxed)
    );
    HttpResponse::Ok()
        .content_type("text/plain; version=0.0.4")
        .body(body)
}

async fn models(state: web::Data<AppState>, request: HttpRequest) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    HttpResponse::Ok().json(json!({
        "object": "list",
        "data": [{
            "id": state.config.model_id,
            "object": "model",
            "created": 0,
            "owned_by": "nerva"
        }]
    }))
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
        reject_streaming(&body)?;
        reject_n_gt_one(&body)?;
        let prompts = completion_prompts(&body)?;
        let created = unix_seconds();
        let id = state.next_response_id("cmpl");
        let mut choices = Vec::with_capacity(prompts.len());
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        for (index, prompt) in prompts.into_iter().enumerate() {
            let generated = generate_text(
                state.clone(),
                GenerateOptions {
                    prompt,
                    prompt_format: PromptFormat::Raw,
                    max_tokens: request_max_tokens(&state, &body)?,
                    temperature: request_f32(&body, "temperature", 1.0)?,
                    top_p: request_f32(&body, "top_p", 1.0)?,
                    top_k: request_u32(&body, "top_k", 0)?,
                    seed: request_u64_opt(&body, "seed")?,
                    stop: request_stop_strings(&body)?,
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
        reject_streaming(&body)?;
        reject_n_gt_one(&body)?;
        let prompt = chat_messages_to_prompt(&body)?;
        let created = unix_seconds();
        let id = state.next_response_id("chatcmpl");
        let generated = generate_text(
            state.clone(),
            GenerateOptions {
                prompt,
                prompt_format: PromptFormat::Auto,
                max_tokens: request_max_tokens(&state, &body)?,
                temperature: request_f32(&body, "temperature", 1.0)?,
                top_p: request_f32(&body, "top_p", 1.0)?,
                top_k: request_u32(&body, "top_k", 0)?,
                seed: request_u64_opt(&body, "seed")?,
                stop: request_stop_strings(&body)?,
            },
        )
        .await?;
        let completion_tokens = generated.token_ids.len();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": state.config.model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated.text
                },
                "logprobs": null,
                "finish_reason": generated.finish_reason
            }],
            "usage": usage(generated.prompt_tokens, completion_tokens)
        })))
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
        reject_streaming(&body)?;
        let prompt = responses_input_to_prompt(&body)?;
        let created = unix_seconds();
        let id = state.next_response_id("resp");
        let generated = generate_text(
            state.clone(),
            GenerateOptions {
                prompt,
                prompt_format: PromptFormat::Auto,
                max_tokens: request_max_tokens(&state, &body)?,
                temperature: request_f32(&body, "temperature", 1.0)?,
                top_p: request_f32(&body, "top_p", 1.0)?,
                top_k: request_u32(&body, "top_k", 0)?,
                seed: request_u64_opt(&body, "seed")?,
                stop: request_stop_strings(&body)?,
            },
        )
        .await?;
        let output_id = state.next_response_id("msg");
        let content_id = state.next_response_id("ct");
        let completion_tokens = generated.token_ids.len();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "model": state.config.model_id,
            "output": [{
                "id": output_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "id": content_id,
                    "type": "output_text",
                    "text": generated.text,
                    "annotations": []
                }]
            }],
            "output_text": generated.text,
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

async fn tokenize(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        require_known_model(&state, &body)?;
        let prompt = string_any(&body, &["prompt", "text", "input"])?
            .ok_or_else(|| ApiError::bad_request("tokenize requires prompt, text, or input"))?;
        let formatted =
            format_prompt_for_model(&state.config.model_path, &prompt, PromptFormat::Raw)
                .map_err(ApiError::bad_request)?;
        let encoded = encode_text_prompt(&state.config.model_path, &formatted.text)
            .map_err(ApiError::bad_request)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "tokens": encoded.token_ids,
            "count": encoded.token_ids.len()
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn detokenize(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        require_known_model(&state, &body)?;
        let tokens = body
            .get("tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| ApiError::bad_request("detokenize requires tokens array"))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|token| u32::try_from(token).ok())
                    .map(TokenId)
                    .ok_or_else(|| ApiError::bad_request("tokens must be u32 token ids"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let text = decode_generated_text(&state.config.model_path, &tokens)
            .map_err(ApiError::bad_request)?
            .ok_or_else(|| ApiError::bad_request("tokenizer decode is unavailable"))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "prompt": text,
            "text": text
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn unsupported_embeddings(request: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    ApiError::unsupported("embeddings require an embedding model backend; this NERVA build serves causal LM text generation only").into_response()
}

async fn unsupported_pooling(request: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    ApiError::unsupported("pooling, classify, score, and rerank require a pooling/ranking backend; this NERVA build serves causal LM text generation only").into_response()
}

async fn unsupported_audio(request: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    ApiError::unsupported("audio transcription and translation require an audio model backend; this NERVA build serves causal LM text generation only").into_response()
}

async fn unsupported_multimodal(request: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    ApiError::unsupported("image generation and image editing require a multimodal/image backend; this NERVA build serves causal LM text generation only").into_response()
}

async fn unsupported_moderations(request: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    ApiError::unsupported("moderations require a moderation model backend; this NERVA build serves causal LM text generation only").into_response()
}

async fn unsupported_batch_api(request: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    ApiError::unsupported("OpenAI batch jobs require persistent request storage and offline scheduling; this NERVA server currently serves online text requests").into_response()
}

async fn unsupported_files_api(request: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    ApiError::unsupported("OpenAI files require a persistent file store; this NERVA server currently serves online text requests").into_response()
}

async fn unsupported_fine_tuning(request: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    ApiError::unsupported("fine tuning jobs are not implemented in the NERVA inference server").into_response()
}

async fn unsupported_lora(request: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    ApiError::unsupported("LoRA adapter hot-load/unload is not implemented for the current NERVA CUDA generation path").into_response()
}

async fn unsupported_admin_state(request: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    ApiError::unsupported("engine sleep, wake, prefix-cache reset, and profiler controls are not implemented for this NERVA server").into_response()
}

async fn is_sleeping(request: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    HttpResponse::Ok().json(json!({"is_sleeping": false}))
}

async fn not_found(request: HttpRequest) -> HttpResponse {
    ApiError::not_found(format!("unknown route: {}", request.path())).into_response()
}

async fn generate_text(
    state: web::Data<AppState>,
    options: GenerateOptions,
) -> Result<GeneratedText, ApiError> {
    let permit = state
        .limiter
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| ApiError::internal("request limiter closed"))?;
    let config = state.config.clone();
    let runtime = state.runtime.clone();
    let result = web::block(move || {
        let _permit = permit;
        generate_text_sync(&runtime, &config, options)
    })
    .await
    .map_err(|err| ApiError::internal(format!("generation task failed: {err}")))?;
    let generated = result?;
    state
        .generated_tokens
        .fetch_add(generated.token_ids.len() as u64, Ordering::Relaxed);
    Ok(generated)
}

fn generate_text_sync(
    runtime: &Runtime,
    config: &ResolvedServeConfig,
    options: GenerateOptions,
) -> Result<GeneratedText, ApiError> {
    if options.max_tokens == 0 {
        return Err(ApiError::bad_request("max_tokens must be non-zero"));
    }
    validate_sampling(options.temperature, options.top_p)?;
    let formatted = format_prompt_for_model(
        &config.model_path,
        &options.prompt,
        options.prompt_format,
    )
    .map_err(ApiError::bad_request)?;
    let encoded =
        encode_text_prompt(&config.model_path, &formatted.text).map_err(ApiError::bad_request)?;
    let prompt_tokens = encoded
        .token_ids
        .iter()
        .copied()
        .map(TokenId)
        .collect::<Vec<_>>();
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
    let output =
        run_hf_causal_lm_cuda_shard_backed_device_generate_with_sampler_profiling_rt_and_progress(
            runtime,
            &config.model_path,
            &prompt_tokens,
            context_tokens,
            options.max_tokens,
            config.queue_capacity,
            config.compute_capability,
            sampler,
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
    let finish_reason = if stopped_by_stop_string || output.stop_reason() == HfCausalLmStopReason::EosToken {
        "stop"
    } else {
        "length"
    };
    Ok(GeneratedText {
        text,
        token_ids,
        prompt_tokens: prompt_tokens.len(),
        finish_reason,
    })
}

fn authorize(state: &AppState, request: &HttpRequest) -> Result<(), ApiError> {
    let Some(api_key) = state.config.api_key.as_deref() else {
        return Ok(());
    };
    let Some(value) = request.headers().get("authorization") else {
        return Err(ApiError::unauthorized());
    };
    let Ok(value) = value.to_str() else {
        return Err(ApiError::unauthorized());
    };
    if value == format!("Bearer {api_key}") {
        Ok(())
    } else {
        Err(ApiError::unauthorized())
    }
}

fn require_known_model(state: &AppState, body: &Value) -> Result<(), ApiError> {
    let Some(model) = body.get("model").and_then(Value::as_str) else {
        return Ok(());
    };
    if model == state.config.model_id {
        Ok(())
    } else {
        Err(ApiError::not_found(format!(
            "model '{model}' is not served by this NERVA instance"
        )))
    }
}

fn reject_streaming(body: &Value) -> Result<(), ApiError> {
    if body.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        Err(ApiError::unsupported(
            "stream=true is not implemented yet in the NERVA OpenAI server",
        ))
    } else {
        Ok(())
    }
}

fn reject_n_gt_one(body: &Value) -> Result<(), ApiError> {
    let n = body.get("n").and_then(Value::as_u64).unwrap_or(1);
    if n == 1 {
        Ok(())
    } else {
        Err(ApiError::unsupported("n > 1 is not implemented yet"))
    }
}

fn completion_prompts(body: &Value) -> Result<Vec<String>, ApiError> {
    match body.get("prompt") {
        Some(Value::String(prompt)) => Ok(vec![prompt.clone()]),
        Some(Value::Array(prompts)) => prompts
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| ApiError::bad_request("prompt arrays must contain strings"))
            })
            .collect(),
        Some(_) => Err(ApiError::bad_request(
            "prompt must be a string or an array of strings",
        )),
        None => Err(ApiError::bad_request("missing prompt")),
    }
}

fn chat_messages_to_prompt(body: &Value) -> Result<String, ApiError> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| ApiError::bad_request("chat completions require messages array"))?;
    if messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    let mut prompt = String::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let content = message_content_text(message.get("content"))?;
        if content.trim().is_empty() {
            continue;
        }
        match role {
            "system" => prompt.push_str("System: "),
            "assistant" => prompt.push_str("Assistant: "),
            "tool" => prompt.push_str("Tool: "),
            _ => prompt.push_str("User: "),
        }
        prompt.push_str(&content);
        prompt.push('\n');
    }
    prompt.push_str("Assistant:");
    Ok(prompt)
}

fn responses_input_to_prompt(body: &Value) -> Result<String, ApiError> {
    let mut prompt = String::new();
    if let Some(instructions) = body.get("instructions").and_then(Value::as_str) {
        if !instructions.trim().is_empty() {
            prompt.push_str("System: ");
            prompt.push_str(instructions);
            prompt.push('\n');
        }
    }
    match body.get("input") {
        Some(Value::String(input)) => prompt.push_str(input),
        Some(Value::Array(messages)) => {
            let body = json!({ "messages": messages });
            prompt.push_str(&chat_messages_to_prompt(&body)?);
        }
        Some(_) => return Err(ApiError::bad_request("input must be a string or messages array")),
        None => return Err(ApiError::bad_request("responses require input")),
    }
    Ok(prompt)
}

fn message_content_text(content: Option<&Value>) -> Result<String, ApiError> {
    match content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") | Some("input_text") | None => {
                        if let Some(value) = part.get("text").and_then(Value::as_str) {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(value);
                        }
                    }
                    Some(other) => {
                        return Err(ApiError::unsupported(format!(
                            "message content part '{other}' is not supported by the text-only backend"
                        )));
                    }
                }
            }
            Ok(text)
        }
        Some(Value::Null) | None => Ok(String::new()),
        Some(_) => Err(ApiError::bad_request(
            "message content must be a string or text content array",
        )),
    }
}

fn request_max_tokens(state: &AppState, body: &Value) -> Result<usize, ApiError> {
    let value = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(Value::as_u64)
        .map(|value| {
            usize::try_from(value).map_err(|_| ApiError::bad_request("max_tokens is too large"))
        })
        .transpose()?
        .unwrap_or(state.config.default_output_tokens);
    if value == 0 {
        Err(ApiError::bad_request("max_tokens must be non-zero"))
    } else {
        Ok(value)
    }
}

fn request_f32(body: &Value, name: &'static str, default: f32) -> Result<f32, ApiError> {
    let value = body
        .get(name)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or(default);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ApiError::bad_request(format!("{name} must be finite")))
    }
}

fn request_u32(body: &Value, name: &'static str, default: u32) -> Result<u32, ApiError> {
    body.get(name)
        .and_then(Value::as_u64)
        .map(|value| {
            u32::try_from(value).map_err(|_| ApiError::bad_request(format!("{name} is too large")))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn request_u64_opt(body: &Value, name: &'static str) -> Result<Option<u64>, ApiError> {
    match body.get(name) {
        Some(Value::Number(number)) => number
            .as_u64()
            .map(Some)
            .ok_or_else(|| ApiError::bad_request(format!("{name} must be a u64"))),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be a u64"))),
    }
}

fn request_stop_strings(body: &Value) -> Result<Vec<String>, ApiError> {
    match body.get("stop") {
        Some(Value::String(stop)) => Ok(vec![stop.clone()]),
        Some(Value::Array(stops)) => stops
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| ApiError::bad_request("stop array must contain strings"))
            })
            .collect(),
        Some(Value::Null) | None => Ok(Vec::new()),
        Some(_) => Err(ApiError::bad_request(
            "stop must be a string or an array of strings",
        )),
    }
}

fn string_any(body: &Value, names: &[&str]) -> Result<Option<String>, ApiError> {
    for name in names {
        if let Some(value) = body.get(*name) {
            return value
                .as_str()
                .map(|value| Some(value.to_string()))
                .ok_or_else(|| ApiError::bad_request(format!("{name} must be a string")));
        }
    }
    Ok(None)
}

fn validate_sampling(temperature: f32, top_p: f32) -> Result<(), ApiError> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(ApiError::bad_request("temperature must be finite and >= 0"));
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(ApiError::bad_request("top_p must be finite and in (0, 1]"));
    }
    Ok(())
}

fn apply_stop_strings(text: String, stops: &[String]) -> (String, bool) {
    let Some(index) = stops
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()
    else {
        return (text, false);
    };
    (text[..index].to_string(), true)
}

fn usage(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    })
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
mod tests {
    use serde_json::json;

    use super::{
        apply_stop_strings, chat_messages_to_prompt, completion_prompts, request_stop_strings,
        responses_input_to_prompt,
    };

    #[test]
    fn parses_string_and_array_completion_prompts() {
        assert_eq!(
            completion_prompts(&json!({"prompt": "hello"})).unwrap(),
            vec!["hello".to_string()]
        );
        assert_eq!(
            completion_prompts(&json!({"prompt": ["a", "b"]})).unwrap(),
            vec!["a".to_string(), "b".to_string()]
        );
        assert!(completion_prompts(&json!({"prompt": [1]})).is_err());
    }

    #[test]
    fn renders_chat_messages_to_text_prompt() {
        let prompt = chat_messages_to_prompt(&json!({
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": [{"type": "text", "text": "hello"}]}
            ]
        }))
        .unwrap();
        assert!(prompt.contains("System: be terse"));
        assert!(prompt.contains("User: hello"));
        assert!(prompt.ends_with("Assistant:"));
    }

    #[test]
    fn parses_responses_input_messages() {
        let prompt = responses_input_to_prompt(&json!({
            "instructions": "be useful",
            "input": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        assert!(prompt.contains("System: be useful"));
        assert!(prompt.contains("User: hello"));
    }

    #[test]
    fn parses_stop_strings() {
        assert_eq!(
            request_stop_strings(&json!({"stop": ["END", "STOP"]})).unwrap(),
            vec!["END".to_string(), "STOP".to_string()]
        );
        let (text, stopped) = apply_stop_strings("hello END world".to_string(), &["END".into()]);
        assert_eq!(text, "hello ");
        assert!(stopped);
    }
}
