use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use actix_web::http::StatusCode;
use actix_web::{App, HttpRequest, HttpResponse, HttpServer, web};
use futures_util::StreamExt;
use nerva_core::types::id::token::TokenId;
use nerva_model::causal_lm::types::HfCausalLmStopReason;
use nerva_model::hf::tokenizer::{
    PromptFormat, decode_generated_text, encode_text_prompt, format_prompt_for_model,
};
use nerva_runtime::engine::hf_cuda_decode::file_backed::generate::{
    HfCudaRtDecodeConfig, HfCudaSamplerConfig,
    run_hf_causal_lm_cuda_shard_backed_device_generate_with_sampler_profiling_rt_and_progress,
};
use nerva_runtime::engine::hf_cuda_decode::file_backed::progress::HfCudaDeviceProgressPhase;
use nerva_runtime::engine::hf_cuda_decode::file_backed::shared_fork_batch::run_hf_causal_lm_cuda_shared_fork_batch_probe;
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};
use serde_json::{Value, json};
use tokio::sync::{Semaphore, mpsc};
use tokio_stream::wrappers::ReceiverStream;

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
    sessions: Mutex<HashMap<String, SessionRecord>>,
    context_cache: Mutex<ContextCacheState>,
    mcp_servers: Mutex<HashMap<String, McpServerRecord>>,
    next_id: AtomicU64,
    request_count: AtomicU64,
    generated_tokens: AtomicU64,
    scheduler_admitted: AtomicU64,
    scheduler_completed: AtomicU64,
    scheduler_active: AtomicU64,
    scheduler_cache_hits: AtomicU64,
    scheduler_cache_misses: AtomicU64,
}

#[derive(Clone, Debug)]
struct GenerateOptions {
    prompt: PromptInput,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    seed: Option<u64>,
    stop: Vec<String>,
    session_id: Option<String>,
    cache_key: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PromptInput {
    Text { text: String, format: PromptFormat },
    TokenIds(Vec<TokenId>),
}

#[derive(Clone, Debug)]
struct GeneratedText {
    text: String,
    token_ids: Vec<u32>,
    prompt_tokens: usize,
    finish_reason: &'static str,
    prompt_hash: u64,
    cache_key: String,
    cache_hit: bool,
    session_id: Option<String>,
}

#[derive(Clone, Debug)]
struct PreparedGeneration {
    prompt_tokens: Vec<TokenId>,
    context_tokens: usize,
    sampler: HfCudaSamplerConfig,
    prompt_hash: u64,
    cache_key: String,
    cache_hit: bool,
    session_id: Option<String>,
}

#[derive(Clone, Debug)]
struct StreamRunStats {
    generated_tokens: usize,
    prompt_tokens: usize,
    prompt_hash: u64,
    cache_key: String,
    cache_hit: bool,
    session_id: Option<String>,
}

#[derive(Clone, Debug)]
struct SessionRecord {
    id: String,
    object: &'static str,
    created: u64,
    updated: u64,
    request_count: u64,
    prompt_tokens: u64,
    generated_tokens: u64,
    last_cache_key: Option<String>,
    last_prompt_hash: Option<u64>,
}

#[derive(Clone, Debug)]
struct ContextCacheEntry {
    key: String,
    prompt_hash: u64,
    prompt_tokens: usize,
    created: u64,
    updated: u64,
    hits: u64,
}

#[derive(Clone, Debug, Default)]
struct ContextCacheState {
    entries: HashMap<String, ContextCacheEntry>,
    hits: u64,
    misses: u64,
}

#[derive(Clone, Debug)]
struct McpServerRecord {
    id: String,
    created: u64,
    updated: u64,
    transport: String,
    endpoint: String,
    status: &'static str,
}

#[derive(Copy, Clone, Debug)]
enum StreamKind {
    Completion,
    ChatCompletion,
    Response,
}

#[derive(Clone, Debug)]
struct StreamMeta {
    id: String,
    created: u64,
    model: String,
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
        sessions: Mutex::new(HashMap::new()),
        context_cache: Mutex::new(ContextCacheState::default()),
        mcp_servers: Mutex::new(HashMap::new()),
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
                .route("/v1/batches", web::post().to(unsupported_batch_api))
                .route("/v1/batches", web::get().to(unsupported_batch_api))
                .route("/v1/batches/{_id}", web::get().to(unsupported_batch_api))
                .route(
                    "/v1/batches/{_id}/cancel",
                    web::post().to(unsupported_batch_api),
                )
                .route("/v1/files", web::post().to(unsupported_files_api))
                .route("/v1/files", web::get().to(unsupported_files_api))
                .route("/v1/files/{_id}", web::get().to(unsupported_files_api))
                .route("/v1/files/{_id}", web::delete().to(unsupported_files_api))
                .route(
                    "/v1/files/{_id}/content",
                    web::get().to(unsupported_files_api),
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
            "nerva_openai_generated_tokens_total {}\n",
            "# TYPE nerva_openai_scheduler_active gauge\n",
            "nerva_openai_scheduler_active {}\n",
            "# TYPE nerva_openai_scheduler_admitted_total counter\n",
            "nerva_openai_scheduler_admitted_total {}\n",
            "# TYPE nerva_openai_scheduler_completed_total counter\n",
            "nerva_openai_scheduler_completed_total {}\n",
            "# TYPE nerva_openai_context_cache_hits_total counter\n",
            "nerva_openai_context_cache_hits_total {}\n",
            "# TYPE nerva_openai_context_cache_misses_total counter\n",
            "nerva_openai_context_cache_misses_total {}\n"
        ),
        state.request_count.load(Ordering::Relaxed),
        state.generated_tokens.load(Ordering::Relaxed),
        state.scheduler_active.load(Ordering::Relaxed),
        state.scheduler_admitted.load(Ordering::Relaxed),
        state.scheduler_completed.load(Ordering::Relaxed),
        state.scheduler_cache_hits.load(Ordering::Relaxed),
        state.scheduler_cache_misses.load(Ordering::Relaxed)
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

async fn model(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    let requested = path.into_inner();
    if requested != state.config.model_id {
        return ApiError::not_found(format!(
            "model '{requested}' is not served by this NERVA instance"
        ))
        .into_response();
    }
    HttpResponse::Ok().json(json!({
        "id": state.config.model_id,
        "object": "model",
        "created": 0,
        "owned_by": "nerva"
    }))
}

async fn create_session(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = body
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| state.next_response_id("sess"));
        let now = unix_seconds();
        let record = SessionRecord {
            id: id.clone(),
            object: "session",
            created: now,
            updated: now,
            request_count: 0,
            prompt_tokens: 0,
            generated_tokens: 0,
            last_cache_key: None,
            last_prompt_hash: None,
        };
        lock_sessions(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(session_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn list_sessions(state: web::Data<AppState>, request: HttpRequest) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let sessions = lock_sessions(&state)?
            .values()
            .map(session_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "list",
            "data": sessions
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn get_session(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let sessions = lock_sessions(&state)?;
        let record = sessions
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("session '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(session_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn delete_session(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let removed = lock_sessions(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "session.deleted",
            "deleted": removed
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn context_cache_status(state: web::Data<AppState>, request: HttpRequest) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let cache = lock_context_cache(&state)?;
        let entries = cache
            .entries
            .values()
            .map(context_cache_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "context_cache",
            "entries": entries,
            "entry_count": cache.entries.len(),
            "hits": cache.hits,
            "misses": cache.misses
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn delete_context_cache(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let key = path.into_inner();
        let removed = lock_context_cache(&state)?.entries.remove(&key).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": key,
            "object": "context_cache.deleted",
            "deleted": removed
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn register_mcp_server(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = body
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| state.next_response_id("mcp"));
        let transport = body
            .get("transport")
            .and_then(Value::as_str)
            .unwrap_or("streamable_http")
            .to_string();
        let endpoint = body
            .get("endpoint")
            .or_else(|| body.get("url"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ApiError::bad_request("MCP server registration requires endpoint or url")
            })?
            .to_string();
        let now = unix_seconds();
        let record = McpServerRecord {
            id: id.clone(),
            created: now,
            updated: now,
            transport,
            endpoint,
            status: "registered",
        };
        lock_mcp_servers(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(mcp_server_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn list_mcp_servers(state: web::Data<AppState>, request: HttpRequest) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let servers = lock_mcp_servers(&state)?
            .values()
            .map(mcp_server_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "list",
            "data": servers
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn delete_mcp_server(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let removed = lock_mcp_servers(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "mcp_server.deleted",
            "deleted": removed
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
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
        let mut choices = Vec::with_capacity(prompts.len());
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
            if n > 1 {
                return Err(ApiError::unsupported(
                    "n > 1 currently requires one prompt with greedy sampler for shared-fork batching",
                ));
            }
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
        reject_unsupported_generation_options(&body)?;
        reject_n_gt_one(&body)?;
        let prompt = chat_messages_to_prompt(&body)?;
        let created = unix_seconds();
        let id = state.next_response_id("chatcmpl");
        let options = GenerateOptions {
            prompt: PromptInput::Text {
                text: prompt,
                format: PromptFormat::Auto,
            },
            max_tokens: request_max_tokens(&state, &body)?,
            temperature: request_f32(&body, "temperature", 1.0)?,
            top_p: request_f32(&body, "top_p", 1.0)?,
            top_k: request_u32(&body, "top_k", 0)?,
            seed: request_u64_opt(&body, "seed")?,
            stop: request_stop_strings(&body)?,
            session_id: request_optional_string(&body, "session_id")?,
            cache_key: request_optional_string(&body, "cache_key")?,
        };
        if request_stream(&body) {
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
        let generated = generate_text(state.clone(), options).await?;
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
        reject_unsupported_generation_options(&body)?;
        let prompt = responses_input_to_prompt(&body)?;
        let created = unix_seconds();
        let id = state.next_response_id("resp");
        let options = GenerateOptions {
            prompt: PromptInput::Text {
                text: prompt,
                format: PromptFormat::Auto,
            },
            max_tokens: request_max_tokens(&state, &body)?,
            temperature: request_f32(&body, "temperature", 1.0)?,
            top_p: request_f32(&body, "top_p", 1.0)?,
            top_k: request_u32(&body, "top_k", 0)?,
            seed: request_u64_opt(&body, "seed")?,
            stop: request_stop_strings(&body)?,
            session_id: request_optional_string(&body, "session_id")?,
            cache_key: request_optional_string(&body, "cache_key")?,
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
        let generated = generate_text(state.clone(), options).await?;
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
    ApiError::unsupported("fine tuning jobs are not implemented in the NERVA inference server")
        .into_response()
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

async fn reset_context_cache(request: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let mut cache = lock_context_cache(&state)?;
        let removed = cache.entries.len();
        cache.entries.clear();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "context_cache.reset",
            "removed": removed
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn not_found(request: HttpRequest) -> HttpResponse {
    ApiError::not_found(format!("unknown route: {}", request.path())).into_response()
}

async fn generate_text_stream(
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
    let mut emitted_text = String::new();
    let mut stopped_by_stop_string = false;
    let mut tx_closed = false;
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
                let delta = text_delta(&emitted_text, &limited_text);
                if !delta.is_empty() && !send_stream_delta(&tx, kind, &meta, delta) {
                    tx_closed = true;
                    return;
                }
                emitted_text = limited_text;
                stopped_by_stop_string = hit_stop;
            },
        )
        .map_err(|err| ApiError::internal(format!("generation failed: {err:?}")))?;
    if !tx_closed {
        let final_text = decode_generated_text(&config.model_path, output.tokens())
            .map_err(ApiError::bad_request)?
            .ok_or_else(|| ApiError::bad_request("tokenizer decode is unavailable"))?;
        let (limited_text, hit_stop) = apply_stop_strings(final_text, &options.stop);
        stopped_by_stop_string |= hit_stop;
        let delta = text_delta(&emitted_text, &limited_text);
        if !delta.is_empty() {
            tx_closed = !send_stream_delta(&tx, kind, &meta, delta);
        }
    }
    if !tx_closed {
        let finish_reason = finish_reason(output.stop_reason(), stopped_by_stop_string).to_string();
        send_stream_final(
            &tx,
            kind,
            &meta,
            &finish_reason,
            prepared.prompt_tokens.len(),
            output.tokens().len(),
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

async fn generate_text(
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

async fn generate_text_batch(
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

fn shared_fork_batch_supported(
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

fn reject_unsupported_generation_options(body: &Value) -> Result<(), ApiError> {
    reject_nonzero_penalty(body, "presence_penalty")?;
    reject_nonzero_penalty(body, "frequency_penalty")?;
    reject_nonempty_field(body, "logit_bias")?;
    reject_present_field(body, "logprobs")?;
    reject_present_field(body, "top_logprobs")?;
    reject_present_field(body, "echo")?;
    reject_present_field(body, "suffix")?;
    reject_present_field(body, "best_of")?;
    reject_nonempty_field(body, "tools")?;
    reject_nonempty_field(body, "functions")?;
    if let Some(tool_choice) = body.get("tool_choice") {
        if tool_choice != "none" {
            return Err(ApiError::unsupported(
                "tool_choice requires tool execution support; MCP/tool execution is not wired into generation yet",
            ));
        }
    }
    if let Some(response_format) = body.get("response_format") {
        let ty = response_format
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("text");
        if ty != "text" {
            return Err(ApiError::unsupported(
                "structured response_format is not implemented yet",
            ));
        }
    }
    Ok(())
}

fn reject_present_field(body: &Value, name: &'static str) -> Result<(), ApiError> {
    if body.get(name).is_some_and(|value| !value.is_null()) {
        Err(ApiError::unsupported(format!(
            "{name} is not implemented by the NERVA OpenAI server yet"
        )))
    } else {
        Ok(())
    }
}

fn reject_nonempty_field(body: &Value, name: &'static str) -> Result<(), ApiError> {
    let Some(value) = body.get(name) else {
        return Ok(());
    };
    let empty = match value {
        Value::Null => true,
        Value::Array(items) => items.is_empty(),
        Value::Object(items) => items.is_empty(),
        _ => false,
    };
    if empty {
        Ok(())
    } else {
        Err(ApiError::unsupported(format!(
            "{name} is not implemented by the NERVA OpenAI server yet"
        )))
    }
}

fn reject_nonzero_penalty(body: &Value, name: &'static str) -> Result<(), ApiError> {
    let Some(value) = body.get(name).and_then(Value::as_f64) else {
        return Ok(());
    };
    if value == 0.0 {
        Ok(())
    } else {
        Err(ApiError::unsupported(format!(
            "{name} is not implemented by the NERVA OpenAI server yet"
        )))
    }
}

fn reject_n_gt_one(body: &Value) -> Result<(), ApiError> {
    let n = request_n(body)?;
    if n == 1 {
        Ok(())
    } else {
        Err(ApiError::unsupported(
            "n > 1 is only implemented for legacy completions in this build",
        ))
    }
}

fn request_n(body: &Value) -> Result<usize, ApiError> {
    let value = body
        .get("n")
        .and_then(Value::as_u64)
        .map(|value| usize::try_from(value).map_err(|_| ApiError::bad_request("n is too large")))
        .transpose()?
        .unwrap_or(1);
    if value == 0 {
        Err(ApiError::bad_request("n must be non-zero"))
    } else {
        Ok(value)
    }
}

fn request_stream(body: &Value) -> bool {
    body.get("stream").and_then(Value::as_bool).unwrap_or(false)
}

fn completion_prompts(body: &Value) -> Result<Vec<PromptInput>, ApiError> {
    match body.get("prompt") {
        Some(Value::String(prompt)) => Ok(vec![text_prompt(prompt.clone(), PromptFormat::Raw)]),
        Some(Value::Array(items)) if items.iter().all(Value::is_number) => {
            Ok(vec![PromptInput::TokenIds(parse_token_id_array(items)?)])
        }
        Some(Value::Array(items)) if items.iter().all(Value::is_string) => items
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(|prompt| text_prompt(prompt.to_string(), PromptFormat::Raw))
                    .ok_or_else(|| ApiError::bad_request("prompt arrays must contain strings"))
            })
            .collect(),
        Some(Value::Array(items)) if items.iter().all(Value::is_array) => items
            .iter()
            .map(|value| {
                let tokens = value.as_array().ok_or_else(|| {
                    ApiError::bad_request("prompt token arrays must contain arrays")
                })?;
                Ok(PromptInput::TokenIds(parse_token_id_array(tokens)?))
            })
            .collect(),
        Some(_) => Err(ApiError::bad_request(
            "prompt must be a string, token-id array, string array, or token-id array array",
        )),
        None => Err(ApiError::bad_request("missing prompt")),
    }
}

fn text_prompt(text: String, format: PromptFormat) -> PromptInput {
    PromptInput::Text { text, format }
}

fn empty_text_prompt() -> PromptInput {
    text_prompt(String::new(), PromptFormat::Raw)
}

fn parse_token_id_array(items: &[Value]) -> Result<Vec<TokenId>, ApiError> {
    if items.is_empty() {
        return Err(ApiError::bad_request(
            "prompt token arrays must not be empty",
        ));
    }
    items
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|token| u32::try_from(token).ok())
                .map(TokenId)
                .ok_or_else(|| ApiError::bad_request("prompt token ids must fit u32"))
        })
        .collect()
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
        Some(_) => {
            return Err(ApiError::bad_request(
                "input must be a string or messages array",
            ));
        }
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

fn request_optional_string(body: &Value, name: &'static str) -> Result<Option<String>, ApiError> {
    match body.get(name) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.clone())),
        Some(Value::String(_)) => Err(ApiError::bad_request(format!("{name} must not be empty"))),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be a string"))),
    }
}

fn record_context_cache_probe(
    state: &AppState,
    key: &str,
    prompt_hash: u64,
    prompt_tokens: usize,
) -> Result<bool, ApiError> {
    let mut cache = lock_context_cache(state)?;
    let now = unix_seconds();
    let hit = match cache.entries.get_mut(key) {
        Some(entry) if entry.prompt_hash == prompt_hash => {
            entry.updated = now;
            entry.hits = entry.hits.saturating_add(1);
            true
        }
        Some(entry) => {
            entry.prompt_hash = prompt_hash;
            entry.prompt_tokens = prompt_tokens;
            entry.updated = now;
            entry.hits = 0;
            false
        }
        None => {
            cache.entries.insert(
                key.to_string(),
                ContextCacheEntry {
                    key: key.to_string(),
                    prompt_hash,
                    prompt_tokens,
                    created: now,
                    updated: now,
                    hits: 0,
                },
            );
            false
        }
    };
    if hit {
        cache.hits = cache.hits.saturating_add(1);
        state.scheduler_cache_hits.fetch_add(1, Ordering::Relaxed);
    } else {
        cache.misses = cache.misses.saturating_add(1);
        state.scheduler_cache_misses.fetch_add(1, Ordering::Relaxed);
    }
    Ok(hit)
}

fn ensure_session_for_request(
    state: &AppState,
    session_id: Option<&str>,
) -> Result<Option<String>, ApiError> {
    let Some(session_id) = session_id else {
        return Ok(None);
    };
    let mut sessions = lock_sessions(state)?;
    if !sessions.contains_key(session_id) {
        let now = unix_seconds();
        sessions.insert(
            session_id.to_string(),
            SessionRecord {
                id: session_id.to_string(),
                object: "session",
                created: now,
                updated: now,
                request_count: 0,
                prompt_tokens: 0,
                generated_tokens: 0,
                last_cache_key: None,
                last_prompt_hash: None,
            },
        );
    }
    Ok(Some(session_id.to_string()))
}

fn record_session_generation(state: &AppState, stats: &StreamRunStats) {
    let Some(session_id) = stats.session_id.as_deref() else {
        return;
    };
    let Ok(mut sessions) = state.sessions.lock() else {
        return;
    };
    if let Some(session) = sessions.get_mut(session_id) {
        session.updated = unix_seconds();
        session.request_count = session.request_count.saturating_add(1);
        session.prompt_tokens = session
            .prompt_tokens
            .saturating_add(stats.prompt_tokens as u64);
        session.generated_tokens = session
            .generated_tokens
            .saturating_add(stats.generated_tokens as u64);
        session.last_cache_key = Some(stats.cache_key.clone());
        session.last_prompt_hash = Some(stats.prompt_hash);
        let _ = stats.cache_hit;
    }
}

fn lock_sessions(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, SessionRecord>>, ApiError> {
    state
        .sessions
        .lock()
        .map_err(|_| ApiError::internal("session registry lock poisoned"))
}

fn lock_context_cache(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, ContextCacheState>, ApiError> {
    state
        .context_cache
        .lock()
        .map_err(|_| ApiError::internal("context cache lock poisoned"))
}

fn lock_mcp_servers(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, McpServerRecord>>, ApiError> {
    state
        .mcp_servers
        .lock()
        .map_err(|_| ApiError::internal("MCP server registry lock poisoned"))
}

fn session_json(record: &SessionRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created": record.created,
        "updated": record.updated,
        "request_count": record.request_count,
        "prompt_tokens": record.prompt_tokens,
        "generated_tokens": record.generated_tokens,
        "last_cache_key": record.last_cache_key,
        "last_prompt_hash": record.last_prompt_hash.map(|hash| format!("{hash:016x}"))
    })
}

fn context_cache_json(entry: &ContextCacheEntry) -> Value {
    json!({
        "id": entry.key,
        "object": "context_cache_entry",
        "prompt_hash": format!("{:016x}", entry.prompt_hash),
        "prompt_tokens": entry.prompt_tokens,
        "created": entry.created,
        "updated": entry.updated,
        "hits": entry.hits
    })
}

fn mcp_server_json(record: &McpServerRecord) -> Value {
    json!({
        "id": record.id,
        "object": "mcp_server",
        "created": record.created,
        "updated": record.updated,
        "transport": record.transport,
        "endpoint": record.endpoint,
        "status": record.status
    })
}

fn hash_tokens(tokens: &[TokenId]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for token in tokens {
        hash ^= u64::from(token.0);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn send_stream_delta(
    tx: &mpsc::Sender<web::Bytes>,
    kind: StreamKind,
    meta: &StreamMeta,
    delta: &str,
) -> bool {
    match kind {
        StreamKind::Completion => send_sse_json(
            tx,
            None,
            json!({
                "id": meta.id,
                "object": "text_completion",
                "created": meta.created,
                "model": meta.model,
                "choices": [{
                    "text": delta,
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": null
                }]
            }),
        ),
        StreamKind::ChatCompletion => send_sse_json(
            tx,
            None,
            json!({
                "id": meta.id,
                "object": "chat.completion.chunk",
                "created": meta.created,
                "model": meta.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": delta
                    },
                    "logprobs": null,
                    "finish_reason": null
                }]
            }),
        ),
        StreamKind::Response => send_sse_json(
            tx,
            Some("response.output_text.delta"),
            json!({
                "type": "response.output_text.delta",
                "response_id": meta.id,
                "item_id": format!("{}-message", meta.id),
                "output_index": 0,
                "content_index": 0,
                "delta": delta
            }),
        ),
    }
}

fn send_stream_final(
    tx: &mpsc::Sender<web::Bytes>,
    kind: StreamKind,
    meta: &StreamMeta,
    finish_reason: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> bool {
    match kind {
        StreamKind::Completion => send_sse_json(
            tx,
            None,
            json!({
                "id": meta.id,
                "object": "text_completion",
                "created": meta.created,
                "model": meta.model,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": finish_reason
                }],
                "usage": usage(prompt_tokens, completion_tokens)
            }),
        ),
        StreamKind::ChatCompletion => send_sse_json(
            tx,
            None,
            json!({
                "id": meta.id,
                "object": "chat.completion.chunk",
                "created": meta.created,
                "model": meta.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "logprobs": null,
                    "finish_reason": finish_reason
                }],
                "usage": usage(prompt_tokens, completion_tokens)
            }),
        ),
        StreamKind::Response => send_sse_json(
            tx,
            Some("response.completed"),
            json!({
                "type": "response.completed",
                "response": {
                    "id": meta.id,
                    "object": "response",
                    "created_at": meta.created,
                    "status": "completed",
                    "model": meta.model,
                    "output": [],
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
            }),
        ),
    }
}

fn send_stream_error(tx: &mpsc::Sender<web::Bytes>, error: ApiError) {
    let _ = send_sse_json(
        tx,
        Some("error"),
        json!({
            "error": {
                "message": error.message,
                "type": error.code,
                "param": null,
                "code": error.code
            }
        }),
    );
    send_stream_done(tx);
}

fn send_stream_done(tx: &mpsc::Sender<web::Bytes>) -> bool {
    tx.blocking_send(web::Bytes::from_static(b"data: [DONE]\n\n"))
        .is_ok()
}

fn send_sse_json(tx: &mpsc::Sender<web::Bytes>, event: Option<&str>, value: Value) -> bool {
    let mut frame = String::new();
    if let Some(event) = event {
        frame.push_str("event: ");
        frame.push_str(event);
        frame.push('\n');
    }
    frame.push_str("data: ");
    frame.push_str(&value.to_string());
    frame.push_str("\n\n");
    tx.blocking_send(web::Bytes::from(frame)).is_ok()
}

fn text_delta<'a>(previous: &str, current: &'a str) -> &'a str {
    current.strip_prefix(previous).unwrap_or(current)
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

fn finish_reason(stop_reason: HfCausalLmStopReason, stopped_by_stop_string: bool) -> &'static str {
    if stopped_by_stop_string || stop_reason == HfCausalLmStopReason::EosToken {
        "stop"
    } else {
        "length"
    }
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
        PromptInput, apply_stop_strings, chat_messages_to_prompt, completion_prompts,
        finish_reason, hash_tokens, reject_unsupported_generation_options, request_n,
        request_optional_string, request_stop_strings, responses_input_to_prompt, session_json,
        shared_fork_batch_supported, text_delta,
    };
    use crate::openai::SessionRecord;
    use nerva_model::causal_lm::types::HfCausalLmStopReason;

    #[test]
    fn parses_string_and_array_completion_prompts() {
        let prompts = completion_prompts(&json!({"prompt": "hello"})).unwrap();
        assert!(matches!(prompts[0], PromptInput::Text { .. }));

        let prompts = completion_prompts(&json!({"prompt": ["a", "b"]})).unwrap();
        assert_eq!(prompts.len(), 2);
        assert!(matches!(prompts[0], PromptInput::Text { .. }));

        let prompts = completion_prompts(&json!({"prompt": [1, 2, 3]})).unwrap();
        assert!(matches!(prompts[0], PromptInput::TokenIds(_)));

        let prompts = completion_prompts(&json!({"prompt": [[1, 2], [3, 4]]})).unwrap();
        assert_eq!(prompts.len(), 2);
        assert!(matches!(prompts[1], PromptInput::TokenIds(_)));
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

    #[test]
    fn computes_stream_text_delta() {
        assert_eq!(text_delta("hello", "hello world"), " world");
        assert_eq!(text_delta("abc", "xyz"), "xyz");
    }

    #[test]
    fn maps_finish_reason_to_openai_values() {
        assert_eq!(finish_reason(HfCausalLmStopReason::EosToken, false), "stop");
        assert_eq!(finish_reason(HfCausalLmStopReason::MaxSteps, true), "stop");
        assert_eq!(
            finish_reason(HfCausalLmStopReason::MaxSteps, false),
            "length"
        );
    }

    #[test]
    fn rejects_unsupported_generation_options() {
        assert!(reject_unsupported_generation_options(&json!({})).is_ok());
        assert!(reject_unsupported_generation_options(&json!({"presence_penalty": 0.0})).is_ok());
        assert!(reject_unsupported_generation_options(&json!({"presence_penalty": 1.0})).is_err());
        assert!(
            reject_unsupported_generation_options(&json!({"tools": [{"type": "mcp"}]})).is_err()
        );
        assert!(
            reject_unsupported_generation_options(&json!({"response_format": {"type": "text"}}))
                .is_ok()
        );
        assert!(
            reject_unsupported_generation_options(
                &json!({"response_format": {"type": "json_object"}})
            )
            .is_err()
        );
    }

    #[test]
    fn parses_n_and_optional_strings() {
        assert_eq!(request_n(&json!({})).unwrap(), 1);
        assert_eq!(request_n(&json!({"n": 3})).unwrap(), 3);
        assert!(request_n(&json!({"n": 0})).is_err());
        assert_eq!(
            request_optional_string(&json!({"session_id": "abc"}), "session_id").unwrap(),
            Some("abc".to_string())
        );
        assert!(request_optional_string(&json!({"session_id": ""}), "session_id").is_err());
    }

    #[test]
    fn hashes_tokens_stably() {
        let a = hash_tokens(&[
            nerva_core::types::id::token::TokenId(1),
            nerva_core::types::id::token::TokenId(2),
        ]);
        let b = hash_tokens(&[
            nerva_core::types::id::token::TokenId(1),
            nerva_core::types::id::token::TokenId(2),
        ]);
        let c = hash_tokens(&[
            nerva_core::types::id::token::TokenId(2),
            nerva_core::types::id::token::TokenId(1),
        ]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn serializes_session_records() {
        let record = SessionRecord {
            id: "sess-1".to_string(),
            object: "session",
            created: 1,
            updated: 2,
            request_count: 3,
            prompt_tokens: 4,
            generated_tokens: 5,
            last_cache_key: Some("cache".to_string()),
            last_prompt_hash: Some(0xabcd),
        };
        let value = session_json(&record);
        assert_eq!(value["id"], "sess-1");
        assert_eq!(value["last_prompt_hash"], "000000000000abcd");
    }

    #[test]
    fn gates_shared_fork_batch_to_greedy_sampler() {
        assert!(shared_fork_batch_supported(0.0, 1.0, 0, None));
        assert!(!shared_fork_batch_supported(1.0, 1.0, 0, None));
        assert!(!shared_fork_batch_supported(0.0, 0.95, 0, None));
        assert!(!shared_fork_batch_supported(0.0, 1.0, 40, None));
        assert!(!shared_fork_batch_supported(0.0, 1.0, 0, Some(7)));
    }
}
