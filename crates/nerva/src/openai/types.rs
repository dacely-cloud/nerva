use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use actix_web::HttpResponse;
use actix_web::http::StatusCode;
use nerva_core::types::id::token::TokenId;
use nerva_model::hf::tokenizer::PromptFormat;
use nerva_runtime::engine::hf_cuda_decode::file_backed::generate::{
    HfCudaRtDecodeConfig, HfCudaSamplerConfig,
};
use nerva_runtime::engine::runtime::Runtime;
use serde_json::{Value, json};
use tokio::sync::Semaphore;

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
pub(crate) struct ResolvedServeConfig {
    pub(crate) model_id: String,
    pub(crate) model_path: String,
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) context_tokens: Option<usize>,
    pub(crate) default_output_tokens: usize,
    pub(crate) queue_capacity: usize,
    pub(crate) compute_capability: Option<u32>,
    pub(crate) max_concurrent_requests: usize,
    pub(crate) workers: Option<usize>,
    pub(crate) max_blocking_threads: Option<usize>,
    pub(crate) api_key: Option<String>,
    pub(crate) rt_decode: HfCudaRtDecodeConfig,
    pub(crate) profiling: bool,
}

pub(crate) struct AppState {
    pub(crate) config: Arc<ResolvedServeConfig>,
    pub(crate) runtime: Runtime,
    pub(crate) limiter: Arc<Semaphore>,
    pub(crate) sessions: Mutex<HashMap<String, SessionRecord>>,
    pub(crate) context_cache: Mutex<ContextCacheState>,
    pub(crate) mcp_servers: Mutex<HashMap<String, McpServerRecord>>,
    pub(crate) files: Mutex<HashMap<String, FileRecord>>,
    pub(crate) batches: Mutex<HashMap<String, BatchRecord>>,
    pub(crate) responses: Mutex<HashMap<String, StoredResponseRecord>>,
    pub(crate) next_id: AtomicU64,
    pub(crate) request_count: AtomicU64,
    pub(crate) generated_tokens: AtomicU64,
    pub(crate) scheduler_admitted: AtomicU64,
    pub(crate) scheduler_completed: AtomicU64,
    pub(crate) scheduler_active: AtomicU64,
    pub(crate) scheduler_cache_hits: AtomicU64,
    pub(crate) scheduler_cache_misses: AtomicU64,
}

#[derive(Clone, Debug)]
pub(crate) struct GenerateOptions {
    pub(crate) prompt: PromptInput,
    pub(crate) max_tokens: usize,
    pub(crate) temperature: f32,
    pub(crate) top_p: f32,
    pub(crate) top_k: u32,
    pub(crate) seed: Option<u64>,
    pub(crate) stop: Vec<String>,
    pub(crate) session_id: Option<String>,
    pub(crate) cache_key: Option<String>,
    pub(crate) output_prefix: Option<String>,
    pub(crate) output_suffix: Option<String>,
    pub(crate) include_reasoning: bool,
    pub(crate) reasoning_mode: ReasoningMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReasoningMode {
    None,
    DeepSeekChat,
    DeepSeekThinking,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PromptInput {
    Text { text: String, format: PromptFormat },
    TokenIds(Vec<TokenId>),
}

#[derive(Clone, Debug)]
pub(crate) struct GeneratedText {
    pub(crate) text: String,
    pub(crate) token_ids: Vec<u32>,
    pub(crate) prompt_tokens: usize,
    pub(crate) finish_reason: &'static str,
    pub(crate) prompt_hash: u64,
    pub(crate) cache_key: String,
    pub(crate) cache_hit: bool,
    pub(crate) session_id: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedGeneration {
    pub(crate) prompt_tokens: Vec<TokenId>,
    pub(crate) context_tokens: usize,
    pub(crate) sampler: HfCudaSamplerConfig,
    pub(crate) prompt_hash: u64,
    pub(crate) cache_key: String,
    pub(crate) cache_hit: bool,
    pub(crate) session_id: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct StreamRunStats {
    pub(crate) generated_tokens: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) prompt_hash: u64,
    pub(crate) cache_key: String,
    pub(crate) cache_hit: bool,
    pub(crate) session_id: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct SessionRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) updated: u64,
    pub(crate) request_count: u64,
    pub(crate) prompt_tokens: u64,
    pub(crate) generated_tokens: u64,
    pub(crate) last_cache_key: Option<String>,
    pub(crate) last_prompt_hash: Option<u64>,
}

#[derive(Clone, Debug)]
pub(crate) struct ContextCacheEntry {
    pub(crate) key: String,
    pub(crate) prompt_hash: u64,
    pub(crate) prompt_tokens: usize,
    pub(crate) created: u64,
    pub(crate) updated: u64,
    pub(crate) hits: u64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ContextCacheState {
    pub(crate) entries: HashMap<String, ContextCacheEntry>,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct McpServerRecord {
    pub(crate) id: String,
    pub(crate) created: u64,
    pub(crate) updated: u64,
    pub(crate) transport: String,
    pub(crate) endpoint: String,
    pub(crate) status: String,
    pub(crate) capabilities: Value,
    pub(crate) tools: Value,
    pub(crate) last_error: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct McpToolInvocation {
    pub(crate) server_id: Option<String>,
    pub(crate) server_label: Option<String>,
    pub(crate) server_url: Option<String>,
    pub(crate) name: String,
    pub(crate) arguments: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct McpToolResult {
    pub(crate) server_id: String,
    pub(crate) name: String,
    pub(crate) arguments: Value,
    pub(crate) result: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct FileRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) bytes: usize,
    pub(crate) created_at: u64,
    pub(crate) filename: String,
    pub(crate) purpose: String,
    pub(crate) status: &'static str,
    pub(crate) content: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(crate) struct BatchRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) endpoint: String,
    pub(crate) input_file_id: String,
    pub(crate) completion_window: String,
    pub(crate) status: String,
    pub(crate) created_at: u64,
    pub(crate) in_progress_at: Option<u64>,
    pub(crate) finalizing_at: Option<u64>,
    pub(crate) completed_at: Option<u64>,
    pub(crate) failed_at: Option<u64>,
    pub(crate) cancelled_at: Option<u64>,
    pub(crate) expires_at: Option<u64>,
    pub(crate) output_file_id: Option<String>,
    pub(crate) error_file_id: Option<String>,
    pub(crate) request_counts: BatchRequestCounts,
    pub(crate) metadata: Value,
    pub(crate) errors: Vec<Value>,
}

#[derive(Clone, Debug)]
pub(crate) struct StoredResponseRecord {
    pub(crate) response: Value,
    pub(crate) input_items: Vec<Value>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BatchRequestCounts {
    pub(crate) total: u64,
    pub(crate) completed: u64,
    pub(crate) failed: u64,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum StreamKind {
    Completion,
    ChatCompletion,
    Response,
}

#[derive(Clone, Debug)]
pub(crate) struct StreamMeta {
    pub(crate) id: String,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) response: Option<ResponseStreamOptions>,
}

#[derive(Clone, Debug)]
pub(crate) struct ResponseStreamOptions {
    pub(crate) store: bool,
    pub(crate) metadata: Value,
    pub(crate) previous_response_id: Option<String>,
    pub(crate) input_items: Vec<Value>,
}

#[derive(Clone, Debug)]
pub(crate) struct ApiError {
    pub(crate) status: StatusCode,
    pub(crate) code: &'static str,
    pub(crate) message: String,
}

impl ApiError {
    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "invalid_request_error",
            message: message.into(),
        }
    }

    pub(crate) fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            code: "not_found_error",
            message: message.into(),
        }
    }

    pub(crate) fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "authentication_error",
            message: "invalid or missing bearer token".to_string(),
        }
    }

    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            code: "unsupported_operation",
            message: message.into(),
        }
    }

    pub(crate) fn bad_gateway(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            code: "upstream_error",
            message: message.into(),
        }
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "server_error",
            message: message.into(),
        }
    }

    pub(crate) fn into_response(self) -> HttpResponse {
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
