use std::collections::HashMap;
use std::sync::MutexGuard;
use std::sync::atomic::Ordering;

use actix_web::{HttpRequest, HttpResponse, web};
use futures_util::stream;
use serde_json::{Map, Value, json};

use crate::cli::args::{DEFAULT_TEMPERATURE, DEFAULT_TOP_P};

use super::{
    ApiError, AppState, AssistantMessageRecord, AssistantRecord, AssistantRunRecord,
    AssistantRunStepRecord, AssistantThreadRecord, GenerateOptions, PromptInput, ReasoningMode,
    apply_response_format_instruction, authorize, chat_prompt_for_reasoning, generate_text,
    percent_decode_query, request_f32, request_max_tokens, request_metadata,
    request_optional_string, request_reasoning_mode, request_response_format_instruction,
    request_stop_strings, request_u32, request_u64_opt, require_known_model,
    split_generated_reasoning, unix_seconds, usage,
};

const ASSISTANT_RUN_TTL_SECONDS: u64 = 10 * 60;

pub(crate) async fn create_assistant(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        let record = assistant_from_body(&state, &body, None)?;
        lock_assistants(&state)?.insert(record.id.clone(), record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_assistants(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let assistants = lock_assistants(&state)?;
        let mut records = assistants.values().cloned().collect::<Vec<_>>();
        sort_by_created(&mut records, query_order(&request));
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "list",
            "data": records.iter().map(assistant_json).collect::<Vec<_>>(),
            "first_id": records.first().map(|record| record.id.as_str()),
            "last_id": records.last().map(|record| record.id.as_str()),
            "has_more": false
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_assistant(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let assistants = lock_assistants(&state)?;
        let record = assistants
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("assistant '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn update_assistant(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        require_known_model(&state, &body)?;
        let id = path.into_inner();
        let mut assistants = lock_assistants(&state)?;
        let record = assistants
            .get_mut(&id)
            .ok_or_else(|| ApiError::not_found(format!("assistant '{id}' does not exist")))?;
        patch_assistant_record(record, &body)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_assistant(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let deleted = lock_assistants(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "assistant.deleted",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_assistant_thread(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        let record = create_thread_record(&state, &body)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_thread_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_thread_and_run(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        let thread_body = body.get("thread").cloned().unwrap_or_else(|| json!({}));
        let thread = create_thread_record(&state, &thread_body)?;
        create_run_response(state.clone(), thread.id, &body).await
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_assistant_thread(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let threads = lock_assistant_threads(&state)?;
        let record = threads
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("thread '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_thread_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn update_assistant_thread(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let mut threads = lock_assistant_threads(&state)?;
        let record = threads
            .get_mut(&id)
            .ok_or_else(|| ApiError::not_found(format!("thread '{id}' does not exist")))?;
        if body.get("metadata").is_some() {
            record.metadata = metadata_or_empty(&body)?;
        }
        if body.get("tool_resources").is_some() {
            record.tool_resources = object_or_empty(&body, "tool_resources")?;
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_thread_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_assistant_thread(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let deleted = lock_assistant_threads(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "thread.deleted",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_assistant_message(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        let thread_id = path.into_inner();
        let mut threads = lock_assistant_threads(&state)?;
        let thread = threads
            .get_mut(&thread_id)
            .ok_or_else(|| ApiError::not_found(format!("thread '{thread_id}' does not exist")))?;
        let record = message_from_body(&state, &thread_id, &body, None, None)?;
        insert_thread_message(thread, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_message_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_assistant_messages(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let thread_id = path.into_inner();
        let threads = lock_assistant_threads(&state)?;
        let thread = threads
            .get(&thread_id)
            .ok_or_else(|| ApiError::not_found(format!("thread '{thread_id}' does not exist")))?;
        let records = ordered_thread_messages(thread, query_order(&request));
        Ok::<_, ApiError>(HttpResponse::Ok().json(list_response(records, assistant_message_json)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_assistant_message(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (thread_id, message_id) = path.into_inner();
        let threads = lock_assistant_threads(&state)?;
        let thread = threads
            .get(&thread_id)
            .ok_or_else(|| ApiError::not_found(format!("thread '{thread_id}' does not exist")))?;
        let record = thread.messages.get(&message_id).ok_or_else(|| {
            ApiError::not_found(format!(
                "message '{message_id}' does not exist in thread '{thread_id}'"
            ))
        })?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_message_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn update_assistant_message(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (thread_id, message_id) = path.into_inner();
        let mut threads = lock_assistant_threads(&state)?;
        let thread = threads
            .get_mut(&thread_id)
            .ok_or_else(|| ApiError::not_found(format!("thread '{thread_id}' does not exist")))?;
        let record = thread.messages.get_mut(&message_id).ok_or_else(|| {
            ApiError::not_found(format!(
                "message '{message_id}' does not exist in thread '{thread_id}'"
            ))
        })?;
        if body.get("metadata").is_some() {
            record.metadata = metadata_or_empty(&body)?;
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_message_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_assistant_message(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (thread_id, message_id) = path.into_inner();
        let mut threads = lock_assistant_threads(&state)?;
        let thread = threads
            .get_mut(&thread_id)
            .ok_or_else(|| ApiError::not_found(format!("thread '{thread_id}' does not exist")))?;
        let deleted = thread.messages.remove(&message_id).is_some();
        if deleted {
            thread.message_order.retain(|id| id != &message_id);
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": message_id,
            "object": "thread.message.deleted",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_assistant_run(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        create_run_response(state.clone(), path.into_inner(), &body).await
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_assistant_runs(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let thread_id = path.into_inner();
        let threads = lock_assistant_threads(&state)?;
        let thread = threads
            .get(&thread_id)
            .ok_or_else(|| ApiError::not_found(format!("thread '{thread_id}' does not exist")))?;
        let records = ordered_thread_runs(thread, query_order(&request));
        Ok::<_, ApiError>(HttpResponse::Ok().json(list_response(records, assistant_run_json)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_assistant_run(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (thread_id, run_id) = path.into_inner();
        let threads = lock_assistant_threads(&state)?;
        let run = thread_run(&threads, &thread_id, &run_id)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_run_json(run)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn update_assistant_run(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (thread_id, run_id) = path.into_inner();
        let mut threads = lock_assistant_threads(&state)?;
        let run = thread_run_mut(&mut threads, &thread_id, &run_id)?;
        if body.get("metadata").is_some() {
            run.metadata = metadata_or_empty(&body)?;
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_run_json(run)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn cancel_assistant_run(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (thread_id, run_id) = path.into_inner();
        let mut threads = lock_assistant_threads(&state)?;
        let run = thread_run_mut(&mut threads, &thread_id, &run_id)?;
        if !is_terminal_run_status(&run.status) {
            run.status = "cancelled".to_string();
            run.cancelled_at = Some(unix_seconds());
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_run_json(run)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn submit_assistant_tool_outputs(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (thread_id, run_id) = path.into_inner();
        let tool_outputs = body
            .get("tool_outputs")
            .and_then(Value::as_array)
            .ok_or_else(|| ApiError::bad_request("submit_tool_outputs requires tool_outputs"))?
            .clone();
        let mut threads = lock_assistant_threads(&state)?;
        let thread = threads
            .get_mut(&thread_id)
            .ok_or_else(|| ApiError::not_found(format!("thread '{thread_id}' does not exist")))?;
        let run = thread.runs.get_mut(&run_id).ok_or_else(|| {
            ApiError::not_found(format!(
                "run '{run_id}' does not exist in thread '{thread_id}'"
            ))
        })?;
        let now = unix_seconds();
        let step = AssistantRunStepRecord {
            id: state.next_response_id("step"),
            object: "thread.run.step",
            created_at: now,
            assistant_id: run.assistant_id.clone(),
            thread_id: thread_id.clone(),
            run_id: run_id.clone(),
            step_type: "tool_calls".to_string(),
            status: "completed".to_string(),
            completed_at: Some(now),
            cancelled_at: None,
            failed_at: None,
            expired_at: None,
            last_error: Value::Null,
            step_details: json!({
                "type": "tool_calls",
                "tool_calls": tool_outputs
            }),
            usage: json!({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        };
        run.steps.push(step);
        run.required_action = Value::Null;
        if !is_terminal_run_status(&run.status) {
            run.status = "completed".to_string();
            run.completed_at = Some(now);
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_run_json(run)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_assistant_run_steps(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (thread_id, run_id) = path.into_inner();
        let threads = lock_assistant_threads(&state)?;
        let run = thread_run(&threads, &thread_id, &run_id)?;
        let mut records = run.steps.iter().collect::<Vec<_>>();
        if query_order(&request) != "asc" {
            records.reverse();
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(list_response(records, assistant_run_step_json)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_assistant_run_step(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (thread_id, run_id, step_id) = path.into_inner();
        let threads = lock_assistant_threads(&state)?;
        let run = thread_run(&threads, &thread_id, &run_id)?;
        let step = run
            .steps
            .iter()
            .find(|step| step.id == step_id)
            .ok_or_else(|| {
                ApiError::not_found(format!(
                    "run step '{step_id}' does not exist in run '{run_id}'"
                ))
            })?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(assistant_run_step_json(step)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn create_run_response(
    state: web::Data<AppState>,
    thread_id: String,
    body: &Value,
) -> Result<HttpResponse, ApiError> {
    let stream_requested = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let execution = execute_assistant_run(state, thread_id, body).await?;
    if stream_requested {
        Ok(assistant_run_sse_response(&execution))
    } else {
        Ok(HttpResponse::Ok().json(execution.final_run))
    }
}

struct AssistantRunExecution {
    initial_run: Value,
    final_run: Value,
    message: Value,
    step: Value,
}

async fn execute_assistant_run(
    state: web::Data<AppState>,
    thread_id: String,
    body: &Value,
) -> Result<AssistantRunExecution, ApiError> {
    let assistant_id = required_string(body, "assistant_id")?;
    let assistant = {
        let assistants = lock_assistants(&state)?;
        assistants.get(&assistant_id).cloned().ok_or_else(|| {
            ApiError::not_found(format!("assistant '{assistant_id}' does not exist"))
        })?
    };
    let run_model =
        request_optional_string(body, "model")?.unwrap_or_else(|| assistant.model.clone());
    require_known_model(&state, &json!({"model": run_model}))?;

    let additional_messages = body
        .get("additional_messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let now = unix_seconds();
    let run_id = state.next_response_id("run");
    let mut run = run_from_body(body, &assistant, &thread_id, &run_id, now, &run_model)?;
    let thread_snapshot = {
        let mut threads = lock_assistant_threads(&state)?;
        let thread = threads
            .get_mut(&thread_id)
            .ok_or_else(|| ApiError::not_found(format!("thread '{thread_id}' does not exist")))?;
        for item in additional_messages {
            let record = message_from_body(&state, &thread_id, &item, None, None)?;
            insert_thread_message(thread, record);
        }
        thread.runs.insert(run_id.clone(), run.clone());
        thread.run_order.push(run_id.clone());
        thread.clone()
    };
    let initial_run = assistant_run_json(&run);

    let prompt = assistant_run_prompt(&state, &assistant, &thread_snapshot, body)?;
    let generated = match generate_text(
        state.clone(),
        GenerateOptions {
            prompt,
            max_tokens: request_max_tokens(&state, body)?,
            temperature: request_f32(body, "temperature", DEFAULT_TEMPERATURE)?,
            top_p: request_f32(body, "top_p", DEFAULT_TOP_P)?,
            top_k: request_u32(body, "top_k", 0)?,
            seed: request_u64_opt(body, "seed")?,
            stop: request_stop_strings(body)?,
            session_id: request_optional_string(body, "session_id")?.or(Some(thread_id.clone())),
            cache_key: request_optional_string(body, "cache_key")?,
            output_prefix: None,
            output_suffix: None,
            include_reasoning: false,
            reasoning_mode: request_reasoning_mode(&state, body)?,
        },
    )
    .await
    {
        Ok(generated) => generated,
        Err(error) => {
            mark_run_failed(&state, &thread_id, &run_id, &error.message)?;
            return Err(error);
        }
    };

    let completed = unix_seconds();
    let split = split_generated_reasoning(&generated.text, ReasoningMode::None);
    let assistant_message = AssistantMessageRecord {
        id: state.next_response_id("msg"),
        object: "thread.message",
        created_at: completed,
        thread_id: thread_id.clone(),
        status: "completed".to_string(),
        incomplete_details: Value::Null,
        completed_at: Some(completed),
        incomplete_at: None,
        role: "assistant".to_string(),
        content: normalize_assistant_message_content(&Value::String(split.content.clone()))?,
        assistant_id: Some(assistant_id.clone()),
        run_id: Some(run_id.clone()),
        attachments: Vec::new(),
        metadata: Value::Object(Map::new()),
    };
    let step = AssistantRunStepRecord {
        id: state.next_response_id("step"),
        object: "thread.run.step",
        created_at: completed,
        assistant_id: assistant_id.clone(),
        thread_id: thread_id.clone(),
        run_id: run_id.clone(),
        step_type: "message_creation".to_string(),
        status: "completed".to_string(),
        completed_at: Some(completed),
        cancelled_at: None,
        failed_at: None,
        expired_at: None,
        last_error: Value::Null,
        step_details: json!({
            "type": "message_creation",
            "message_creation": {
                "message_id": assistant_message.id
            }
        }),
        usage: usage(generated.prompt_tokens, generated.token_ids.len()),
    };
    run.status = "completed".to_string();
    run.completed_at = Some(completed);
    run.expires_at = None;
    run.usage = usage(generated.prompt_tokens, generated.token_ids.len());
    run.steps.push(step.clone());
    let final_run = {
        let mut threads = lock_assistant_threads(&state)?;
        let thread = threads
            .get_mut(&thread_id)
            .ok_or_else(|| ApiError::not_found(format!("thread '{thread_id}' does not exist")))?;
        insert_thread_message(thread, assistant_message.clone());
        let stored_run = thread.runs.get_mut(&run_id).ok_or_else(|| {
            ApiError::not_found(format!(
                "run '{run_id}' does not exist in thread '{thread_id}'"
            ))
        })?;
        *stored_run = run.clone();
        assistant_run_json(stored_run)
    };
    Ok(AssistantRunExecution {
        initial_run,
        final_run,
        message: assistant_message_json(&assistant_message),
        step: assistant_run_step_json(&step),
    })
}

fn create_thread_record(state: &AppState, body: &Value) -> Result<AssistantThreadRecord, ApiError> {
    let now = unix_seconds();
    let id = state.next_response_id("thread");
    let mut record = AssistantThreadRecord {
        id: id.clone(),
        object: "thread",
        created_at: now,
        metadata: metadata_or_empty(body)?,
        tool_resources: object_or_empty(body, "tool_resources")?,
        messages: HashMap::new(),
        message_order: Vec::new(),
        runs: HashMap::new(),
        run_order: Vec::new(),
    };
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for item in messages {
            let message = message_from_body(state, &id, item, None, None)?;
            insert_thread_message(&mut record, message);
        }
    }
    lock_assistant_threads(state)?.insert(id, record.clone());
    Ok(record)
}

fn assistant_from_body(
    state: &AppState,
    body: &Value,
    id: Option<String>,
) -> Result<AssistantRecord, ApiError> {
    let model = required_string(body, "model")?;
    require_known_model(state, &json!({"model": model}))?;
    Ok(AssistantRecord {
        id: id.unwrap_or_else(|| state.next_response_id("asst")),
        object: "assistant",
        created_at: unix_seconds(),
        name: optional_string_field(body, "name")?,
        description: optional_string_field(body, "description")?,
        model,
        instructions: optional_string_field(body, "instructions")?,
        tools: array_or_empty(body, "tools")?,
        tool_resources: object_or_empty(body, "tool_resources")?,
        metadata: metadata_or_empty(body)?,
        temperature: optional_f64_field(body, "temperature")?,
        top_p: optional_f64_field(body, "top_p")?,
        response_format: body
            .get("response_format")
            .cloned()
            .unwrap_or_else(|| json!("auto")),
        reasoning_effort: optional_string_field(body, "reasoning_effort")?,
    })
}

fn patch_assistant_record(record: &mut AssistantRecord, body: &Value) -> Result<(), ApiError> {
    if body.get("model").is_some() {
        record.model = required_string(body, "model")?;
    }
    if body.get("name").is_some() {
        record.name = optional_string_field(body, "name")?;
    }
    if body.get("description").is_some() {
        record.description = optional_string_field(body, "description")?;
    }
    if body.get("instructions").is_some() {
        record.instructions = optional_string_field(body, "instructions")?;
    }
    if body.get("tools").is_some() {
        record.tools = array_or_empty(body, "tools")?;
    }
    if body.get("tool_resources").is_some() {
        record.tool_resources = object_or_empty(body, "tool_resources")?;
    }
    if body.get("metadata").is_some() {
        record.metadata = metadata_or_empty(body)?;
    }
    if body.get("temperature").is_some() {
        record.temperature = optional_f64_field(body, "temperature")?;
    }
    if body.get("top_p").is_some() {
        record.top_p = optional_f64_field(body, "top_p")?;
    }
    if let Some(response_format) = body.get("response_format") {
        record.response_format = response_format.clone();
    }
    if body.get("reasoning_effort").is_some() {
        record.reasoning_effort = optional_string_field(body, "reasoning_effort")?;
    }
    Ok(())
}

fn message_from_body(
    state: &AppState,
    thread_id: &str,
    body: &Value,
    assistant_id: Option<String>,
    run_id: Option<String>,
) -> Result<AssistantMessageRecord, ApiError> {
    let role = required_string(body, "role")?;
    if !matches!(role.as_str(), "user" | "assistant") {
        return Err(ApiError::bad_request(
            "assistant message role must be user or assistant",
        ));
    }
    let content = body
        .get("content")
        .ok_or_else(|| ApiError::bad_request("assistant message requires content"))?;
    let now = unix_seconds();
    Ok(AssistantMessageRecord {
        id: state.next_response_id("msg"),
        object: "thread.message",
        created_at: now,
        thread_id: thread_id.to_string(),
        status: "completed".to_string(),
        incomplete_details: Value::Null,
        completed_at: Some(now),
        incomplete_at: None,
        role,
        content: normalize_assistant_message_content(content)?,
        assistant_id,
        run_id,
        attachments: array_or_empty(body, "attachments")?,
        metadata: metadata_or_empty(body)?,
    })
}

fn run_from_body(
    body: &Value,
    assistant: &AssistantRecord,
    thread_id: &str,
    run_id: &str,
    now: u64,
    model: &str,
) -> Result<AssistantRunRecord, ApiError> {
    let instructions = optional_string_field(body, "instructions")?.or_else(|| {
        let additional = optional_string_field(body, "additional_instructions")
            .ok()
            .flatten();
        match (assistant.instructions.clone(), additional) {
            (Some(base), Some(extra)) if !extra.is_empty() => Some(format!("{base}\n{extra}")),
            (base, _) => base,
        }
    });
    Ok(AssistantRunRecord {
        id: run_id.to_string(),
        object: "thread.run",
        created_at: now,
        thread_id: thread_id.to_string(),
        assistant_id: assistant.id.clone(),
        status: "in_progress".to_string(),
        started_at: Some(now),
        expires_at: Some(now.saturating_add(ASSISTANT_RUN_TTL_SECONDS)),
        cancelled_at: None,
        failed_at: None,
        completed_at: None,
        required_action: Value::Null,
        last_error: Value::Null,
        incomplete_details: Value::Null,
        model: model.to_string(),
        instructions,
        tools: if body.get("tools").is_some() {
            array_or_empty(body, "tools")?
        } else {
            assistant.tools.clone()
        },
        tool_resources: body
            .get("tool_resources")
            .cloned()
            .unwrap_or_else(|| assistant.tool_resources.clone()),
        metadata: metadata_or_empty(body)?,
        temperature: optional_f64_field(body, "temperature")?.or(assistant.temperature),
        top_p: optional_f64_field(body, "top_p")?.or(assistant.top_p),
        response_format: body
            .get("response_format")
            .cloned()
            .unwrap_or_else(|| assistant.response_format.clone()),
        parallel_tool_calls: optional_bool_field(body, "parallel_tool_calls")?.unwrap_or(true),
        max_prompt_tokens: optional_u64_field(body, "max_prompt_tokens")?,
        max_completion_tokens: optional_u64_field(body, "max_completion_tokens")?,
        usage: Value::Null,
        steps: Vec::new(),
    })
}

pub(crate) fn assistant_run_chat_body(
    assistant: &AssistantRecord,
    thread: &AssistantThreadRecord,
    body: &Value,
) -> Value {
    let mut messages = Vec::new();
    let instructions = body
        .get("instructions")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            let additional = body
                .get("additional_instructions")
                .and_then(Value::as_str)
                .map(str::to_string);
            match (assistant.instructions.clone(), additional) {
                (Some(base), Some(extra)) if !extra.trim().is_empty() => {
                    Some(format!("{base}\n{extra}"))
                }
                (base, _) => base,
            }
        });
    if let Some(instructions) = instructions
        && !instructions.trim().is_empty()
    {
        messages.push(json!({"role": "system", "content": instructions}));
    }
    for message_id in &thread.message_order {
        let Some(message) = thread.messages.get(message_id) else {
            continue;
        };
        let text = assistant_message_text(&message.content);
        if !text.trim().is_empty() {
            messages.push(json!({
                "role": message.role,
                "content": text
            }));
        }
    }
    if !messages
        .iter()
        .any(|message| message.get("role").and_then(Value::as_str) != Some("system"))
    {
        messages.push(json!({"role": "user", "content": "Continue."}));
    }
    json!({ "messages": messages })
}

fn assistant_run_prompt(
    state: &AppState,
    assistant: &AssistantRecord,
    thread: &AssistantThreadRecord,
    body: &Value,
) -> Result<PromptInput, ApiError> {
    let chat_body = assistant_run_chat_body(assistant, thread, body);
    let reasoning_mode = request_reasoning_mode(state, body)?;
    let prompt = chat_prompt_for_reasoning(&chat_body, reasoning_mode)?;
    apply_response_format_instruction(
        prompt,
        request_response_format_instruction(body)?.as_deref(),
    )
}

pub(crate) fn normalize_assistant_message_content(content: &Value) -> Result<Vec<Value>, ApiError> {
    match content {
        Value::String(text) => Ok(vec![assistant_text_block(text)]),
        Value::Array(items) => items
            .iter()
            .map(|item| match item {
                Value::Object(object) => match object.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = object
                            .get("text")
                            .and_then(|value| {
                                value
                                    .as_str()
                                    .or_else(|| value.get("value").and_then(Value::as_str))
                            })
                            .unwrap_or("");
                        Ok(assistant_text_block(text))
                    }
                    Some("image_file") | Some("image_url") => Ok(item.clone()),
                    Some(other) => Err(ApiError::unsupported(format!(
                        "assistant message content part '{other}' is not implemented"
                    ))),
                    None => Err(ApiError::bad_request(
                        "assistant message content part requires type",
                    )),
                },
                _ => Err(ApiError::bad_request(
                    "assistant message content parts must be objects",
                )),
            })
            .collect(),
        _ => Err(ApiError::bad_request(
            "assistant message content must be a string or content part array",
        )),
    }
}

fn assistant_text_block(text: &str) -> Value {
    json!({
        "type": "text",
        "text": {
            "value": text,
            "annotations": []
        }
    })
}

pub(crate) fn assistant_message_text(content: &[Value]) -> String {
    let mut out = String::new();
    for part in content {
        let text = part
            .get("text")
            .and_then(|value| {
                value
                    .as_str()
                    .or_else(|| value.get("value").and_then(Value::as_str))
            })
            .unwrap_or("");
        if !out.is_empty() && !text.is_empty() {
            out.push('\n');
        }
        out.push_str(text);
    }
    out
}

fn insert_thread_message(thread: &mut AssistantThreadRecord, record: AssistantMessageRecord) {
    thread.message_order.push(record.id.clone());
    thread.messages.insert(record.id.clone(), record);
}

fn mark_run_failed(
    state: &AppState,
    thread_id: &str,
    run_id: &str,
    message: &str,
) -> Result<(), ApiError> {
    let now = unix_seconds();
    let mut threads = lock_assistant_threads(state)?;
    let run = thread_run_mut(&mut threads, thread_id, run_id)?;
    run.status = "failed".to_string();
    run.failed_at = Some(now);
    run.last_error = json!({
        "code": "server_error",
        "message": message
    });
    Ok(())
}

fn assistant_run_sse_response(execution: &AssistantRunExecution) -> HttpResponse {
    let frames = vec![
        sse_json_frame("thread.run.created", execution.initial_run.clone()),
        sse_json_frame("thread.message.completed", execution.message.clone()),
        sse_json_frame("thread.run.step.completed", execution.step.clone()),
        sse_json_frame("thread.run.completed", execution.final_run.clone()),
        "data: [DONE]\n\n".to_string(),
    ];
    HttpResponse::Ok()
        .insert_header(("cache-control", "no-cache"))
        .content_type("text/event-stream")
        .streaming(stream::iter(frames.into_iter().map(|frame| {
            Ok::<web::Bytes, actix_web::Error>(web::Bytes::from(frame))
        })))
}

fn sse_json_frame(event: &str, payload: Value) -> String {
    format!("event: {event}\ndata: {payload}\n\n")
}

pub(crate) fn assistant_json(record: &AssistantRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "name": record.name,
        "description": record.description,
        "model": record.model,
        "instructions": record.instructions,
        "tools": record.tools,
        "tool_resources": record.tool_resources,
        "metadata": record.metadata,
        "temperature": record.temperature,
        "top_p": record.top_p,
        "response_format": record.response_format,
        "reasoning_effort": record.reasoning_effort
    })
}

pub(crate) fn assistant_thread_json(record: &AssistantThreadRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "metadata": record.metadata,
        "tool_resources": record.tool_resources
    })
}

pub(crate) fn assistant_message_json(record: &AssistantMessageRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "thread_id": record.thread_id,
        "status": record.status,
        "incomplete_details": record.incomplete_details,
        "completed_at": record.completed_at,
        "incomplete_at": record.incomplete_at,
        "role": record.role,
        "content": record.content,
        "assistant_id": record.assistant_id,
        "run_id": record.run_id,
        "attachments": record.attachments,
        "metadata": record.metadata
    })
}

pub(crate) fn assistant_run_json(record: &AssistantRunRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "thread_id": record.thread_id,
        "assistant_id": record.assistant_id,
        "status": record.status,
        "started_at": record.started_at,
        "expires_at": record.expires_at,
        "cancelled_at": record.cancelled_at,
        "failed_at": record.failed_at,
        "completed_at": record.completed_at,
        "required_action": record.required_action,
        "last_error": record.last_error,
        "incomplete_details": record.incomplete_details,
        "model": record.model,
        "instructions": record.instructions,
        "tools": record.tools,
        "tool_resources": record.tool_resources,
        "metadata": record.metadata,
        "temperature": record.temperature,
        "top_p": record.top_p,
        "response_format": record.response_format,
        "parallel_tool_calls": record.parallel_tool_calls,
        "max_prompt_tokens": record.max_prompt_tokens,
        "max_completion_tokens": record.max_completion_tokens,
        "usage": record.usage
    })
}

pub(crate) fn assistant_run_step_json(record: &AssistantRunStepRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "assistant_id": record.assistant_id,
        "thread_id": record.thread_id,
        "run_id": record.run_id,
        "type": record.step_type,
        "status": record.status,
        "completed_at": record.completed_at,
        "cancelled_at": record.cancelled_at,
        "failed_at": record.failed_at,
        "expired_at": record.expired_at,
        "last_error": record.last_error,
        "step_details": record.step_details,
        "usage": record.usage
    })
}

fn list_response<T>(records: Vec<&T>, render: fn(&T) -> Value) -> Value {
    json!({
        "object": "list",
        "data": records.iter().map(|record| render(record)).collect::<Vec<_>>(),
        "first_id": records.first().and_then(|record| rendered_id(render(record))),
        "last_id": records.last().and_then(|record| rendered_id(render(record))),
        "has_more": false
    })
}

fn rendered_id(value: Value) -> Option<String> {
    value.get("id").and_then(Value::as_str).map(str::to_string)
}

fn ordered_thread_messages<'a>(
    thread: &'a AssistantThreadRecord,
    order: &str,
) -> Vec<&'a AssistantMessageRecord> {
    ordered_records(&thread.message_order, &thread.messages, order)
}

fn ordered_thread_runs<'a>(
    thread: &'a AssistantThreadRecord,
    order: &str,
) -> Vec<&'a AssistantRunRecord> {
    ordered_records(&thread.run_order, &thread.runs, order)
}

fn ordered_records<'a, T>(
    ids: &[String],
    records: &'a HashMap<String, T>,
    order: &str,
) -> Vec<&'a T> {
    let iter: Box<dyn Iterator<Item = &String> + '_> = if order == "asc" {
        Box::new(ids.iter())
    } else {
        Box::new(ids.iter().rev())
    };
    iter.filter_map(|id| records.get(id)).collect()
}

fn sort_by_created(records: &mut [AssistantRecord], order: &str) {
    records.sort_by_key(|record| record.created_at);
    if order != "asc" {
        records.reverse();
    }
}

fn query_order(request: &HttpRequest) -> &str {
    query_param(request.query_string(), "order")
        .filter(|value| value == "asc")
        .map(|_| "asc")
        .unwrap_or("desc")
}

fn query_param(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|part| {
        let (key, value) = part.split_once('=')?;
        (percent_decode_query(key) == name).then(|| percent_decode_query(value))
    })
}

fn thread_run<'a>(
    threads: &'a MutexGuard<'_, HashMap<String, AssistantThreadRecord>>,
    thread_id: &str,
    run_id: &str,
) -> Result<&'a AssistantRunRecord, ApiError> {
    let thread = threads
        .get(thread_id)
        .ok_or_else(|| ApiError::not_found(format!("thread '{thread_id}' does not exist")))?;
    thread.runs.get(run_id).ok_or_else(|| {
        ApiError::not_found(format!(
            "run '{run_id}' does not exist in thread '{thread_id}'"
        ))
    })
}

fn thread_run_mut<'a>(
    threads: &'a mut MutexGuard<'_, HashMap<String, AssistantThreadRecord>>,
    thread_id: &str,
    run_id: &str,
) -> Result<&'a mut AssistantRunRecord, ApiError> {
    let thread = threads
        .get_mut(thread_id)
        .ok_or_else(|| ApiError::not_found(format!("thread '{thread_id}' does not exist")))?;
    thread.runs.get_mut(run_id).ok_or_else(|| {
        ApiError::not_found(format!(
            "run '{run_id}' does not exist in thread '{thread_id}'"
        ))
    })
}

fn is_terminal_run_status(status: &str) -> bool {
    matches!(
        status,
        "completed" | "failed" | "cancelled" | "expired" | "incomplete"
    )
}

fn required_string(body: &Value, name: &'static str) -> Result<String, ApiError> {
    match body.get(name) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(value.clone()),
        Some(Value::String(_)) => Err(ApiError::bad_request(format!("{name} must not be empty"))),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be a string"))),
        None => Err(ApiError::bad_request(format!("{name} is required"))),
    }
}

fn optional_string_field(body: &Value, name: &'static str) -> Result<Option<String>, ApiError> {
    match body.get(name) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.clone())),
        Some(Value::String(_)) => Ok(None),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be a string"))),
    }
}

fn optional_f64_field(body: &Value, name: &'static str) -> Result<Option<f64>, ApiError> {
    match body.get(name) {
        Some(Value::Number(number)) => number
            .as_f64()
            .filter(|value| value.is_finite())
            .map(Some)
            .ok_or_else(|| ApiError::bad_request(format!("{name} must be finite"))),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be a number"))),
    }
}

fn optional_u64_field(body: &Value, name: &'static str) -> Result<Option<u64>, ApiError> {
    match body.get(name) {
        Some(Value::Number(number)) => number
            .as_u64()
            .map(Some)
            .ok_or_else(|| ApiError::bad_request(format!("{name} must be a u64"))),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be a u64"))),
    }
}

fn optional_bool_field(body: &Value, name: &'static str) -> Result<Option<bool>, ApiError> {
    match body.get(name) {
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be a boolean"))),
    }
}

fn array_or_empty(body: &Value, name: &'static str) -> Result<Vec<Value>, ApiError> {
    match body.get(name) {
        Some(Value::Array(items)) => Ok(items.clone()),
        Some(Value::Null) | None => Ok(Vec::new()),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be an array"))),
    }
}

fn object_or_empty(body: &Value, name: &'static str) -> Result<Value, ApiError> {
    match body.get(name) {
        Some(Value::Object(_)) => Ok(body.get(name).cloned().unwrap_or_else(|| json!({}))),
        Some(Value::Null) | None => Ok(json!({})),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be an object"))),
    }
}

fn metadata_or_empty(body: &Value) -> Result<Value, ApiError> {
    match request_metadata(body)? {
        Value::Null => Ok(json!({})),
        value => Ok(value),
    }
}

fn lock_assistants(
    state: &AppState,
) -> Result<MutexGuard<'_, HashMap<String, AssistantRecord>>, ApiError> {
    state
        .assistants
        .lock()
        .map_err(|_| ApiError::internal("assistant registry lock poisoned"))
}

fn lock_assistant_threads(
    state: &AppState,
) -> Result<MutexGuard<'_, HashMap<String, AssistantThreadRecord>>, ApiError> {
    state
        .assistant_threads
        .lock()
        .map_err(|_| ApiError::internal("assistant thread registry lock poisoned"))
}
