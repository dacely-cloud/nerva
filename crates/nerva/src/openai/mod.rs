use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use actix_web::{App, HttpServer, web};
use nerva_runtime::engine::hf_cuda_decode::file_backed::generate::HfCudaRtDecodeConfig;
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};
use tokio::sync::Semaphore;

use crate::cli::args::{DEFAULT_OUTPUT_TOKENS, DEFAULT_QUEUE_CAPACITY};
use crate::cli::model::{detect_cuda_compute_capability, resolve_model_path};
use crate::json::json_escape;

mod admin;
mod assistants;
mod audio;
mod batches;
mod chat_store;
mod chatkit;
mod containers;
mod context_cache;
mod conversations;
mod deepseek_prompt;
mod embeddings;
mod endpoints;
mod evals;
mod files;
mod fine_tuning;
mod generation;
mod images;
mod mcp;
mod model_catalog;
mod moderations;
mod organization;
mod realtime;
mod requests;
mod response_store;
mod response_tokens;
mod sessions;
mod skills;
mod streaming;
mod types;
mod uploads;
mod vector_stores;
mod videos;
mod voice;

pub(crate) use admin::*;
pub(crate) use assistants::*;
pub(crate) use audio::*;
pub(crate) use batches::*;
pub(crate) use chat_store::*;
pub(crate) use chatkit::*;
pub(crate) use containers::*;
pub(crate) use context_cache::*;
pub(crate) use conversations::*;
pub(crate) use deepseek_prompt::*;
pub(crate) use embeddings::*;
pub(crate) use endpoints::*;
pub(crate) use evals::*;
pub(crate) use files::*;
pub(crate) use fine_tuning::*;
pub(crate) use generation::*;
pub(crate) use images::*;
pub(crate) use mcp::*;
pub(crate) use model_catalog::*;
pub(crate) use moderations::*;
pub(crate) use organization::*;
pub(crate) use realtime::*;
pub(crate) use requests::*;
pub(crate) use response_store::*;
pub(crate) use response_tokens::*;
pub(crate) use sessions::*;
pub(crate) use skills::*;
pub(crate) use streaming::*;
pub(crate) use types::*;
pub(crate) use uploads::*;
pub(crate) use vector_stores::*;
pub(crate) use videos::*;
pub(crate) use voice::*;

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
        realtime_sessions: Mutex::new(HashMap::new()),
        realtime_calls: Mutex::new(HashMap::new()),
        context_cache: Mutex::new(ContextCacheState::default()),
        mcp_servers: Mutex::new(HashMap::new()),
        files: Mutex::new(HashMap::new()),
        audio_voices: Mutex::new(HashMap::new()),
        voice_consents: Mutex::new(HashMap::new()),
        uploads: Mutex::new(HashMap::new()),
        vector_stores: Mutex::new(HashMap::new()),
        containers: Mutex::new(HashMap::new()),
        skills: Mutex::new(HashMap::new()),
        assistants: Mutex::new(HashMap::new()),
        assistant_threads: Mutex::new(HashMap::new()),
        chatkit_sessions: Mutex::new(HashMap::new()),
        chatkit_threads: Mutex::new(HashMap::new()),
        evals: Mutex::new(HashMap::new()),
        fine_tuning_jobs: Mutex::new(HashMap::new()),
        videos: Mutex::new(HashMap::new()),
        video_characters: Mutex::new(HashMap::new()),
        batches: Mutex::new(HashMap::new()),
        responses: Mutex::new(HashMap::new()),
        conversations: Mutex::new(HashMap::new()),
        chat_completions: Mutex::new(HashMap::new()),
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
                .app_data(web::PayloadConfig::new(64 * 1024 * 1024))
                .route("/health", web::get().to(health))
                .route("/version", web::get().to(version))
                .route("/metrics", web::get().to(metrics))
                .route("/tokenize", web::post().to(tokenize))
                .route("/detokenize", web::post().to(detokenize))
                .route("/v1/tokenize", web::post().to(tokenize))
                .route("/v1/detokenize", web::post().to(detokenize))
                .route("/v1/models", web::get().to(models))
                .route("/v1/models/{model}", web::get().to(model))
                .route("/v1/models/{model}", web::delete().to(delete_model))
                .route(
                    "/v1/organization/usage/completions",
                    web::get().to(organization_usage_completions),
                )
                .route(
                    "/v1/organization/usage/embeddings",
                    web::get().to(organization_usage_embeddings),
                )
                .route(
                    "/v1/organization/usage/moderations",
                    web::get().to(organization_usage_moderations),
                )
                .route(
                    "/v1/organization/usage/images",
                    web::get().to(organization_usage_images),
                )
                .route(
                    "/v1/organization/usage/audio_speeches",
                    web::get().to(organization_usage_audio_speeches),
                )
                .route(
                    "/v1/organization/usage/audio_transcriptions",
                    web::get().to(organization_usage_audio_transcriptions),
                )
                .route(
                    "/v1/organization/usage/vector_stores",
                    web::get().to(organization_usage_vector_stores),
                )
                .route(
                    "/v1/organization/usage/code_interpreter_sessions",
                    web::get().to(organization_usage_code_interpreter_sessions),
                )
                .route(
                    "/v1/organization/usage/file_search_calls",
                    web::get().to(organization_usage_file_search_calls),
                )
                .route(
                    "/v1/organization/usage/web_search_calls",
                    web::get().to(organization_usage_web_search_calls),
                )
                .route("/v1/organization/costs", web::get().to(organization_costs))
                .route(
                    "/v1/chatkit/sessions",
                    web::post().to(create_chatkit_session),
                )
                .route(
                    "/v1/chatkit/sessions/{session_id}/cancel",
                    web::post().to(cancel_chatkit_session),
                )
                .route("/v1/chatkit/threads", web::get().to(list_chatkit_threads))
                .route(
                    "/v1/chatkit/threads/{thread_id}",
                    web::get().to(get_chatkit_thread),
                )
                .route(
                    "/v1/chatkit/threads/{thread_id}",
                    web::delete().to(delete_chatkit_thread),
                )
                .route(
                    "/v1/chatkit/threads/{thread_id}/items",
                    web::get().to(list_chatkit_thread_items),
                )
                .route("/v1/assistants", web::post().to(create_assistant))
                .route("/v1/assistants", web::get().to(list_assistants))
                .route(
                    "/v1/assistants/{assistant_id}",
                    web::get().to(get_assistant),
                )
                .route(
                    "/v1/assistants/{assistant_id}",
                    web::post().to(update_assistant),
                )
                .route(
                    "/v1/assistants/{assistant_id}",
                    web::delete().to(delete_assistant),
                )
                .route("/v1/threads", web::post().to(create_assistant_thread))
                .route("/v1/threads/runs", web::post().to(create_thread_and_run))
                .route(
                    "/v1/threads/{thread_id}",
                    web::get().to(get_assistant_thread),
                )
                .route(
                    "/v1/threads/{thread_id}",
                    web::post().to(update_assistant_thread),
                )
                .route(
                    "/v1/threads/{thread_id}",
                    web::delete().to(delete_assistant_thread),
                )
                .route(
                    "/v1/threads/{thread_id}/messages",
                    web::post().to(create_assistant_message),
                )
                .route(
                    "/v1/threads/{thread_id}/messages",
                    web::get().to(list_assistant_messages),
                )
                .route(
                    "/v1/threads/{thread_id}/messages/{message_id}",
                    web::get().to(get_assistant_message),
                )
                .route(
                    "/v1/threads/{thread_id}/messages/{message_id}",
                    web::post().to(update_assistant_message),
                )
                .route(
                    "/v1/threads/{thread_id}/messages/{message_id}",
                    web::delete().to(delete_assistant_message),
                )
                .route(
                    "/v1/threads/{thread_id}/runs",
                    web::post().to(create_assistant_run),
                )
                .route(
                    "/v1/threads/{thread_id}/runs",
                    web::get().to(list_assistant_runs),
                )
                .route(
                    "/v1/threads/{thread_id}/runs/{run_id}",
                    web::get().to(get_assistant_run),
                )
                .route(
                    "/v1/threads/{thread_id}/runs/{run_id}",
                    web::post().to(update_assistant_run),
                )
                .route(
                    "/v1/threads/{thread_id}/runs/{run_id}/cancel",
                    web::post().to(cancel_assistant_run),
                )
                .route(
                    "/v1/threads/{thread_id}/runs/{run_id}/submit_tool_outputs",
                    web::post().to(submit_assistant_tool_outputs),
                )
                .route(
                    "/v1/threads/{thread_id}/runs/{run_id}/steps",
                    web::get().to(list_assistant_run_steps),
                )
                .route(
                    "/v1/threads/{thread_id}/runs/{run_id}/steps/{step_id}",
                    web::get().to(get_assistant_run_step),
                )
                .route("/v1/sessions", web::post().to(create_session))
                .route("/v1/sessions", web::get().to(list_sessions))
                .route("/v1/sessions/{session_id}", web::get().to(get_session))
                .route(
                    "/v1/sessions/{session_id}",
                    web::delete().to(delete_session),
                )
                .route(
                    "/v1/realtime/client_secrets",
                    web::post().to(create_realtime_client_secret),
                )
                .route(
                    "/v1/realtime/sessions",
                    web::post().to(create_realtime_session),
                )
                .route(
                    "/v1/realtime/transcription_sessions",
                    web::post().to(create_realtime_transcription_session),
                )
                .route(
                    "/v1/realtime/translations/client_secrets",
                    web::post().to(create_realtime_translation_client_secret),
                )
                .route(
                    "/v1/realtime/calls/{call_id}/accept",
                    web::post().to(accept_realtime_call),
                )
                .route(
                    "/v1/realtime/calls/{call_id}/reject",
                    web::post().to(reject_realtime_call),
                )
                .route(
                    "/v1/realtime/calls/{call_id}/hangup",
                    web::post().to(hangup_realtime_call),
                )
                .route(
                    "/v1/realtime/calls/{call_id}/refer",
                    web::post().to(refer_realtime_call),
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
                .route("/v1/chat/completions", web::get().to(list_chat_completions))
                .route(
                    "/v1/chat/completions/{completion_id}",
                    web::get().to(get_chat_completion),
                )
                .route(
                    "/v1/chat/completions/{completion_id}",
                    web::post().to(update_chat_completion),
                )
                .route(
                    "/v1/chat/completions/{completion_id}",
                    web::delete().to(delete_chat_completion),
                )
                .route(
                    "/v1/chat/completions/{completion_id}/messages",
                    web::get().to(list_chat_completion_messages),
                )
                .route("/v1/responses", web::post().to(responses))
                .route(
                    "/v1/responses/input_tokens",
                    web::post().to(count_response_input_tokens),
                )
                .route("/v1/responses/compact", web::post().to(compact_response))
                .route("/v1/responses/{response_id}", web::get().to(get_response))
                .route(
                    "/v1/responses/{response_id}",
                    web::delete().to(delete_response),
                )
                .route(
                    "/v1/responses/{response_id}/cancel",
                    web::post().to(cancel_response),
                )
                .route(
                    "/v1/responses/{response_id}/compact",
                    web::post().to(compact_response_by_id),
                )
                .route(
                    "/v1/responses/{response_id}/input_items",
                    web::get().to(list_response_input_items),
                )
                .route("/v1/conversations", web::post().to(create_conversation))
                .route(
                    "/v1/conversations/{conversation_id}",
                    web::get().to(get_conversation),
                )
                .route(
                    "/v1/conversations/{conversation_id}",
                    web::post().to(update_conversation),
                )
                .route(
                    "/v1/conversations/{conversation_id}",
                    web::delete().to(delete_conversation),
                )
                .route(
                    "/v1/conversations/{conversation_id}/items",
                    web::post().to(create_conversation_items),
                )
                .route(
                    "/v1/conversations/{conversation_id}/items",
                    web::get().to(list_conversation_items),
                )
                .route(
                    "/v1/conversations/{conversation_id}/items/{item_id}",
                    web::get().to(get_conversation_item),
                )
                .route(
                    "/v1/conversations/{conversation_id}/items/{item_id}",
                    web::delete().to(delete_conversation_item),
                )
                .route("/v1/embeddings", web::post().to(embeddings))
                .route("/pooling", web::post().to(unsupported_pooling))
                .route("/classify", web::post().to(unsupported_pooling))
                .route("/score", web::post().to(unsupported_pooling))
                .route("/v1/score", web::post().to(unsupported_pooling))
                .route("/rerank", web::post().to(unsupported_pooling))
                .route("/v1/rerank", web::post().to(unsupported_pooling))
                .route("/v2/rerank", web::post().to(unsupported_pooling))
                .route(
                    "/v1/audio/transcriptions",
                    web::post().to(create_audio_transcription),
                )
                .route(
                    "/v1/audio/translations",
                    web::post().to(create_audio_translation),
                )
                .route("/v1/audio/speech", web::post().to(create_audio_speech))
                .route("/v1/audio/voices", web::post().to(create_audio_voice))
                .route(
                    "/v1/audio/voice_consents",
                    web::post().to(create_voice_consent),
                )
                .route(
                    "/v1/audio/voice_consents",
                    web::get().to(list_voice_consents),
                )
                .route(
                    "/v1/audio/voice_consents/{consent_id}",
                    web::get().to(get_voice_consent),
                )
                .route(
                    "/v1/audio/voice_consents/{consent_id}",
                    web::post().to(update_voice_consent),
                )
                .route(
                    "/v1/audio/voice_consents/{consent_id}",
                    web::delete().to(delete_voice_consent),
                )
                .route(
                    "/v1/images/generations",
                    web::post().to(create_image_generation),
                )
                .route("/v1/images/edits", web::post().to(create_image_edit))
                .route(
                    "/v1/images/variations",
                    web::post().to(create_image_variation),
                )
                .route("/v1/videos", web::post().to(create_video))
                .route("/v1/videos", web::get().to(list_videos))
                .route("/v1/videos/edits", web::post().to(edit_video))
                .route("/v1/videos/extensions", web::post().to(extend_video))
                .route(
                    "/v1/videos/characters",
                    web::post().to(create_video_character),
                )
                .route(
                    "/v1/videos/characters/{character_id}",
                    web::get().to(get_video_character),
                )
                .route("/v1/videos/{video_id}", web::get().to(get_video))
                .route("/v1/videos/{video_id}", web::delete().to(delete_video))
                .route("/v1/videos/{video_id}/remix", web::post().to(remix_video))
                .route(
                    "/v1/videos/{video_id}/content",
                    web::get().to(get_video_content),
                )
                .route("/v1/moderations", web::post().to(moderations))
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
                .route("/v1/uploads", web::post().to(create_upload))
                .route(
                    "/v1/uploads/{upload_id}/parts",
                    web::post().to(create_upload_part),
                )
                .route(
                    "/v1/uploads/{upload_id}/complete",
                    web::post().to(complete_upload),
                )
                .route(
                    "/v1/uploads/{upload_id}/cancel",
                    web::post().to(cancel_upload),
                )
                .route("/v1/vector_stores", web::post().to(create_vector_store))
                .route("/v1/vector_stores", web::get().to(list_vector_stores))
                .route(
                    "/v1/vector_stores/{vector_store_id}",
                    web::get().to(get_vector_store),
                )
                .route(
                    "/v1/vector_stores/{vector_store_id}",
                    web::post().to(update_vector_store),
                )
                .route(
                    "/v1/vector_stores/{vector_store_id}",
                    web::delete().to(delete_vector_store),
                )
                .route(
                    "/v1/vector_stores/{vector_store_id}/search",
                    web::post().to(search_vector_store),
                )
                .route(
                    "/v1/vector_stores/{vector_store_id}/files",
                    web::post().to(create_vector_store_file),
                )
                .route(
                    "/v1/vector_stores/{vector_store_id}/files",
                    web::get().to(list_vector_store_files),
                )
                .route(
                    "/v1/vector_stores/{vector_store_id}/files/{file_id}",
                    web::get().to(get_vector_store_file),
                )
                .route(
                    "/v1/vector_stores/{vector_store_id}/files/{file_id}",
                    web::post().to(update_vector_store_file),
                )
                .route(
                    "/v1/vector_stores/{vector_store_id}/files/{file_id}",
                    web::delete().to(delete_vector_store_file),
                )
                .route(
                    "/v1/vector_stores/{vector_store_id}/files/{file_id}/content",
                    web::get().to(get_vector_store_file_content),
                )
                .route(
                    "/v1/vector_stores/{vector_store_id}/file_batches",
                    web::post().to(create_vector_store_file_batch),
                )
                .route(
                    "/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}",
                    web::get().to(get_vector_store_file_batch),
                )
                .route(
                    "/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
                    web::get().to(list_vector_store_file_batch_files),
                )
                .route(
                    "/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel",
                    web::post().to(cancel_vector_store_file_batch),
                )
                .route("/v1/containers", web::post().to(create_container))
                .route("/v1/containers", web::get().to(list_containers))
                .route(
                    "/v1/containers/{container_id}",
                    web::get().to(get_container),
                )
                .route(
                    "/v1/containers/{container_id}",
                    web::delete().to(delete_container),
                )
                .route(
                    "/v1/containers/{container_id}/files",
                    web::post().to(create_container_file),
                )
                .route(
                    "/v1/containers/{container_id}/files",
                    web::get().to(list_container_files),
                )
                .route(
                    "/v1/containers/{container_id}/files/{file_id}",
                    web::get().to(get_container_file),
                )
                .route(
                    "/v1/containers/{container_id}/files/{file_id}",
                    web::delete().to(delete_container_file),
                )
                .route(
                    "/v1/containers/{container_id}/files/{file_id}/content",
                    web::get().to(get_container_file_content),
                )
                .route("/v1/skills", web::post().to(create_skill))
                .route("/v1/skills", web::get().to(list_skills))
                .route("/v1/skills/{skill_id}", web::get().to(get_skill))
                .route("/v1/skills/{skill_id}", web::post().to(update_skill))
                .route("/v1/skills/{skill_id}", web::delete().to(delete_skill))
                .route(
                    "/v1/skills/{skill_id}/content",
                    web::get().to(get_skill_content),
                )
                .route(
                    "/v1/skills/{skill_id}/versions",
                    web::post().to(create_skill_version),
                )
                .route(
                    "/v1/skills/{skill_id}/versions",
                    web::get().to(list_skill_versions),
                )
                .route(
                    "/v1/skills/{skill_id}/versions/{version_id}",
                    web::get().to(get_skill_version),
                )
                .route(
                    "/v1/skills/{skill_id}/versions/{version_id}",
                    web::delete().to(delete_skill_version),
                )
                .route(
                    "/v1/skills/{skill_id}/versions/{version_id}/content",
                    web::get().to(get_skill_version_content),
                )
                .route("/v1/evals", web::post().to(create_eval))
                .route("/v1/evals", web::get().to(list_evals))
                .route("/v1/evals/{eval_id}", web::get().to(get_eval))
                .route("/v1/evals/{eval_id}", web::post().to(update_eval))
                .route("/v1/evals/{eval_id}", web::delete().to(delete_eval))
                .route("/v1/evals/{eval_id}/runs", web::post().to(create_eval_run))
                .route("/v1/evals/{eval_id}/runs", web::get().to(list_eval_runs))
                .route(
                    "/v1/evals/{eval_id}/runs/{run_id}",
                    web::get().to(get_eval_run),
                )
                .route(
                    "/v1/evals/{eval_id}/runs/{run_id}",
                    web::post().to(update_eval_run),
                )
                .route(
                    "/v1/evals/{eval_id}/runs/{run_id}",
                    web::delete().to(delete_eval_run),
                )
                .route(
                    "/v1/evals/{eval_id}/runs/{run_id}/cancel",
                    web::post().to(cancel_eval_run),
                )
                .route(
                    "/v1/evals/{eval_id}/runs/{run_id}/output_items",
                    web::get().to(list_eval_run_output_items),
                )
                .route(
                    "/v1/evals/{eval_id}/runs/{run_id}/output_items/{output_item_id}",
                    web::get().to(get_eval_run_output_item),
                )
                .route(
                    "/v1/fine_tuning/jobs",
                    web::post().to(create_fine_tuning_job),
                )
                .route("/v1/fine_tuning/jobs", web::get().to(list_fine_tuning_jobs))
                .route(
                    "/v1/fine_tuning/jobs/{job_id}",
                    web::get().to(get_fine_tuning_job),
                )
                .route(
                    "/v1/fine_tuning/jobs/{job_id}/events",
                    web::get().to(list_fine_tuning_events),
                )
                .route(
                    "/v1/fine_tuning/jobs/{job_id}/cancel",
                    web::post().to(cancel_fine_tuning_job),
                )
                .route(
                    "/v1/fine_tuning/jobs/{job_id}/pause",
                    web::post().to(pause_fine_tuning_job),
                )
                .route(
                    "/v1/fine_tuning/jobs/{job_id}/resume",
                    web::post().to(resume_fine_tuning_job),
                )
                .route(
                    "/v1/fine_tuning/jobs/{job_id}/checkpoints",
                    web::get().to(list_fine_tuning_checkpoints),
                )
                .route(
                    "/v1/fine_tuning/checkpoints/{checkpoint_id}/permissions",
                    web::post().to(create_fine_tuning_checkpoint_permissions),
                )
                .route(
                    "/v1/fine_tuning/checkpoints/{checkpoint_id}/permissions",
                    web::get().to(list_fine_tuning_checkpoint_permissions),
                )
                .route(
                    "/v1/fine_tuning/checkpoints/{checkpoint_id}/permissions/{permission_id}",
                    web::get().to(get_fine_tuning_checkpoint_permission),
                )
                .route(
                    "/v1/fine_tuning/checkpoints/{checkpoint_id}/permissions/{permission_id}",
                    web::delete().to(delete_fine_tuning_checkpoint_permission),
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
