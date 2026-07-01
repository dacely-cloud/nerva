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
    pub(crate) realtime_sessions: Mutex<HashMap<String, RealtimeSessionRecord>>,
    pub(crate) realtime_calls: Mutex<HashMap<String, RealtimeCallRecord>>,
    pub(crate) context_cache: Mutex<ContextCacheState>,
    pub(crate) mcp_servers: Mutex<HashMap<String, McpServerRecord>>,
    pub(crate) files: Mutex<HashMap<String, FileRecord>>,
    pub(crate) audio_voices: Mutex<HashMap<String, AudioVoiceRecord>>,
    pub(crate) voice_consents: Mutex<HashMap<String, VoiceConsentRecord>>,
    pub(crate) uploads: Mutex<HashMap<String, UploadRecord>>,
    pub(crate) vector_stores: Mutex<HashMap<String, VectorStoreRecord>>,
    pub(crate) containers: Mutex<HashMap<String, ContainerRecord>>,
    pub(crate) skills: Mutex<HashMap<String, SkillRecord>>,
    pub(crate) assistants: Mutex<HashMap<String, AssistantRecord>>,
    pub(crate) assistant_threads: Mutex<HashMap<String, AssistantThreadRecord>>,
    pub(crate) chatkit_sessions: Mutex<HashMap<String, ChatKitSessionRecord>>,
    pub(crate) chatkit_threads: Mutex<HashMap<String, ChatKitThreadRecord>>,
    pub(crate) evals: Mutex<HashMap<String, EvalRecord>>,
    pub(crate) fine_tuning_jobs: Mutex<HashMap<String, FineTuningJobRecord>>,
    pub(crate) videos: Mutex<HashMap<String, VideoRecord>>,
    pub(crate) video_characters: Mutex<HashMap<String, VideoCharacterRecord>>,
    pub(crate) batches: Mutex<HashMap<String, BatchRecord>>,
    pub(crate) responses: Mutex<HashMap<String, StoredResponseRecord>>,
    pub(crate) conversations: Mutex<HashMap<String, ConversationRecord>>,
    pub(crate) chat_completions: Mutex<HashMap<String, StoredChatCompletionRecord>>,
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
pub(crate) struct RealtimeClientSecretRecord {
    pub(crate) value: String,
    pub(crate) expires_at: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct RealtimeSessionRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) kind: &'static str,
    pub(crate) created_at: u64,
    pub(crate) expires_at: u64,
    pub(crate) client_secret: RealtimeClientSecretRecord,
    pub(crate) config: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct RealtimeCallRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) updated_at: u64,
    pub(crate) status: String,
    pub(crate) session_id: Option<String>,
    pub(crate) status_code: Option<u16>,
    pub(crate) target_uri: Option<String>,
    pub(crate) config: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct VideoRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) completed_at: Option<u64>,
    pub(crate) expires_at: Option<u64>,
    pub(crate) model: String,
    pub(crate) prompt: String,
    pub(crate) seconds: String,
    pub(crate) size: String,
    pub(crate) status: String,
    pub(crate) progress: u64,
    pub(crate) operation: String,
    pub(crate) remixed_from_video_id: Option<String>,
    pub(crate) character_id: Option<String>,
    pub(crate) metadata: Value,
    pub(crate) error: Option<Value>,
    pub(crate) content: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(crate) struct VideoCharacterRecord {
    pub(crate) id: String,
    pub(crate) created_at: u64,
    pub(crate) name: String,
    pub(crate) metadata: Value,
    pub(crate) source_video_id: Option<String>,
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
    pub(crate) message_endpoint: Option<String>,
    pub(crate) command: Option<String>,
    pub(crate) args: Vec<String>,
    pub(crate) cwd: Option<String>,
    pub(crate) env: HashMap<String, String>,
    pub(crate) authorization: Option<String>,
    pub(crate) protocol_version: String,
    pub(crate) session_id: Option<String>,
    pub(crate) connector_id: Option<String>,
    pub(crate) allowed_tools: Vec<String>,
    pub(crate) require_approval: Value,
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
    pub(crate) connector_id: Option<String>,
    pub(crate) authorization: Option<String>,
    pub(crate) allowed_tools: Vec<String>,
    pub(crate) require_approval: Value,
    pub(crate) name: String,
    pub(crate) arguments: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct McpToolResult {
    pub(crate) server_id: String,
    pub(crate) server_label: Option<String>,
    pub(crate) name: String,
    pub(crate) arguments: Value,
    pub(crate) result: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct McpApprovalRequest {
    pub(crate) id: String,
    pub(crate) server_id: String,
    pub(crate) server_label: String,
    pub(crate) name: String,
    pub(crate) arguments: Value,
}

#[derive(Clone, Debug)]
pub(crate) enum McpToolExecution {
    Completed(McpToolResult),
    ApprovalRequired(McpApprovalRequest),
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
pub(crate) struct AudioVoiceRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) name: String,
    pub(crate) consent: String,
    pub(crate) sample_filename: String,
    pub(crate) sample_content_type: String,
    pub(crate) sample_bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(crate) struct VoiceConsentRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) language: String,
    pub(crate) name: String,
    pub(crate) recording_filename: String,
    pub(crate) recording_content_type: String,
    pub(crate) recording_bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(crate) struct UploadRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) bytes: u64,
    pub(crate) created_at: u64,
    pub(crate) expires_at: u64,
    pub(crate) filename: String,
    pub(crate) purpose: String,
    pub(crate) mime_type: String,
    pub(crate) status: String,
    pub(crate) parts: Vec<UploadPartRecord>,
    pub(crate) file_id: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct UploadPartRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) upload_id: String,
    pub(crate) bytes: usize,
    pub(crate) content: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(crate) struct VectorStoreRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) name: Option<String>,
    pub(crate) metadata: Value,
    pub(crate) status: String,
    pub(crate) expires_after: Value,
    pub(crate) expires_at: Option<u64>,
    pub(crate) last_active_at: Option<u64>,
    pub(crate) files: HashMap<String, VectorStoreFileRecord>,
    pub(crate) file_batches: HashMap<String, VectorStoreFileBatchRecord>,
}

#[derive(Clone, Debug)]
pub(crate) struct VectorStoreFileRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) vector_store_id: String,
    pub(crate) status: String,
    pub(crate) last_error: Option<Value>,
    pub(crate) usage_bytes: usize,
    pub(crate) attributes: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct VectorStoreFileBatchRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) vector_store_id: String,
    pub(crate) status: String,
    pub(crate) file_ids: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VectorStoreFileCounts {
    pub(crate) in_progress: u64,
    pub(crate) completed: u64,
    pub(crate) failed: u64,
    pub(crate) cancelled: u64,
    pub(crate) total: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct ContainerRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) name: Option<String>,
    pub(crate) status: String,
    pub(crate) metadata: Value,
    pub(crate) expires_after: Value,
    pub(crate) last_active_at: Option<u64>,
    pub(crate) files: HashMap<String, ContainerFileRecord>,
}

#[derive(Clone, Debug)]
pub(crate) struct ContainerFileRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) container_id: String,
    pub(crate) filename: String,
    pub(crate) path: String,
    pub(crate) bytes: usize,
    pub(crate) source_file_id: Option<String>,
    pub(crate) content: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(crate) struct SkillRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) updated_at: u64,
    pub(crate) name: String,
    pub(crate) description: Option<String>,
    pub(crate) metadata: Value,
    pub(crate) status: String,
    pub(crate) current_version_id: Option<String>,
    pub(crate) versions: HashMap<String, SkillVersionRecord>,
}

#[derive(Clone, Debug)]
pub(crate) struct SkillVersionRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) skill_id: String,
    pub(crate) name: String,
    pub(crate) description: Option<String>,
    pub(crate) status: String,
    pub(crate) content_type: String,
    pub(crate) content: Vec<u8>,
    pub(crate) metadata: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct AssistantRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) name: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) model: String,
    pub(crate) instructions: Option<String>,
    pub(crate) tools: Vec<Value>,
    pub(crate) tool_resources: Value,
    pub(crate) metadata: Value,
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) response_format: Value,
    pub(crate) reasoning_effort: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct AssistantThreadRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) metadata: Value,
    pub(crate) tool_resources: Value,
    pub(crate) messages: HashMap<String, AssistantMessageRecord>,
    pub(crate) message_order: Vec<String>,
    pub(crate) runs: HashMap<String, AssistantRunRecord>,
    pub(crate) run_order: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct AssistantMessageRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) thread_id: String,
    pub(crate) status: String,
    pub(crate) incomplete_details: Value,
    pub(crate) completed_at: Option<u64>,
    pub(crate) incomplete_at: Option<u64>,
    pub(crate) role: String,
    pub(crate) content: Vec<Value>,
    pub(crate) assistant_id: Option<String>,
    pub(crate) run_id: Option<String>,
    pub(crate) attachments: Vec<Value>,
    pub(crate) metadata: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct AssistantRunRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) thread_id: String,
    pub(crate) assistant_id: String,
    pub(crate) status: String,
    pub(crate) started_at: Option<u64>,
    pub(crate) expires_at: Option<u64>,
    pub(crate) cancelled_at: Option<u64>,
    pub(crate) failed_at: Option<u64>,
    pub(crate) completed_at: Option<u64>,
    pub(crate) required_action: Value,
    pub(crate) last_error: Value,
    pub(crate) incomplete_details: Value,
    pub(crate) model: String,
    pub(crate) instructions: Option<String>,
    pub(crate) tools: Vec<Value>,
    pub(crate) tool_resources: Value,
    pub(crate) metadata: Value,
    pub(crate) temperature: Option<f64>,
    pub(crate) top_p: Option<f64>,
    pub(crate) response_format: Value,
    pub(crate) parallel_tool_calls: bool,
    pub(crate) max_prompt_tokens: Option<u64>,
    pub(crate) max_completion_tokens: Option<u64>,
    pub(crate) usage: Value,
    pub(crate) steps: Vec<AssistantRunStepRecord>,
}

#[derive(Clone, Debug)]
pub(crate) struct AssistantRunStepRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) assistant_id: String,
    pub(crate) thread_id: String,
    pub(crate) run_id: String,
    pub(crate) step_type: String,
    pub(crate) status: String,
    pub(crate) completed_at: Option<u64>,
    pub(crate) cancelled_at: Option<u64>,
    pub(crate) failed_at: Option<u64>,
    pub(crate) expired_at: Option<u64>,
    pub(crate) last_error: Value,
    pub(crate) step_details: Value,
    pub(crate) usage: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct ChatKitSessionRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) expires_at: u64,
    pub(crate) cancelled_at: Option<u64>,
    pub(crate) status: String,
    pub(crate) client_secret: String,
    pub(crate) workflow: Value,
    pub(crate) scope: Value,
    pub(crate) user: Option<String>,
    pub(crate) chatkit_configuration: Value,
    pub(crate) rate_limits: Value,
    pub(crate) max_requests_per_1_minute: Option<u64>,
    pub(crate) max_requests_per_session: Option<u64>,
    pub(crate) ttl_seconds: u64,
    pub(crate) thread_id: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct ChatKitThreadRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) title: Option<String>,
    pub(crate) user: Option<String>,
    pub(crate) status: Value,
    pub(crate) items: Vec<ChatKitThreadItemRecord>,
    pub(crate) metadata: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct ChatKitThreadItemRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) item_type: String,
    pub(crate) content: Vec<Value>,
    pub(crate) attachments: Vec<Value>,
    pub(crate) metadata: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct EvalRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) updated_at: u64,
    pub(crate) name: String,
    pub(crate) data_source_config: Value,
    pub(crate) testing_criteria: Value,
    pub(crate) metadata: Value,
    pub(crate) status: String,
    pub(crate) runs: HashMap<String, EvalRunRecord>,
}

#[derive(Clone, Debug)]
pub(crate) struct EvalRunRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) eval_id: String,
    pub(crate) status: String,
    pub(crate) data_source: Value,
    pub(crate) metadata: Value,
    pub(crate) result_counts: EvalRunResultCounts,
    pub(crate) output_items: HashMap<String, EvalRunOutputItemRecord>,
}

#[derive(Clone, Debug)]
pub(crate) struct EvalRunOutputItemRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) eval_id: String,
    pub(crate) run_id: String,
    pub(crate) status: String,
    pub(crate) item: Value,
    pub(crate) sample: Value,
    pub(crate) results: Vec<Value>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct EvalRunResultCounts {
    pub(crate) total: u64,
    pub(crate) errored: u64,
    pub(crate) failed: u64,
    pub(crate) passed: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct FineTuningJobRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) finished_at: Option<u64>,
    pub(crate) model: String,
    pub(crate) fine_tuned_model: Option<String>,
    pub(crate) training_file: String,
    pub(crate) validation_file: Option<String>,
    pub(crate) status: String,
    pub(crate) hyperparameters: Value,
    pub(crate) method: Value,
    pub(crate) integrations: Value,
    pub(crate) seed: Option<u64>,
    pub(crate) suffix: Option<String>,
    pub(crate) trained_tokens: Option<u64>,
    pub(crate) estimated_finish: Option<u64>,
    pub(crate) result_files: Vec<String>,
    pub(crate) metadata: Value,
    pub(crate) error: Value,
    pub(crate) events: Vec<FineTuningJobEventRecord>,
    pub(crate) checkpoints: Vec<FineTuningJobCheckpointRecord>,
}

#[derive(Clone, Debug)]
pub(crate) struct FineTuningJobEventRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) level: String,
    pub(crate) message: String,
    pub(crate) data: Value,
    pub(crate) event_type: String,
}

#[derive(Clone, Debug)]
pub(crate) struct FineTuningJobCheckpointRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) fine_tuned_model_checkpoint: String,
    pub(crate) fine_tuning_job_id: String,
    pub(crate) step_number: u64,
    pub(crate) metrics: Value,
    pub(crate) permissions: Vec<FineTuningCheckpointPermissionRecord>,
}

#[derive(Clone, Debug)]
pub(crate) struct FineTuningCheckpointPermissionRecord {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created_at: u64,
    pub(crate) project_id: String,
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

#[derive(Clone, Debug)]
pub(crate) struct ConversationRecord {
    pub(crate) id: String,
    pub(crate) created_at: u64,
    pub(crate) updated_at: u64,
    pub(crate) metadata: Value,
    pub(crate) items: Vec<Value>,
}

#[derive(Clone, Debug)]
pub(crate) struct StoredChatCompletionRecord {
    pub(crate) response: Value,
    pub(crate) messages: Vec<Value>,
    pub(crate) metadata: Value,
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
    pub(crate) conversation_id: Option<String>,
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
