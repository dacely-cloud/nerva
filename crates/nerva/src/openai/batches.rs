use std::collections::HashMap;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use crate::cli::args::{DEFAULT_TEMPERATURE, DEFAULT_TOP_P};

use super::{
    ApiError, AppState, BatchRecord, BatchRequestCounts, GenerateOptions, ImageTask, PromptInput,
    ReasoningMode, VideoOperation, append_response_to_conversation,
    apply_response_format_instruction, authorize, chat_prompt_for_reasoning,
    completion_echo_prefix, completion_prompts, conversation_context,
    create_audio_text_batch_response, create_embeddings_response, create_moderation_response,
    create_speech_batch_response, create_video_batch_response, empty_text_prompt,
    generate_text_batch_direct_sync, generate_text_direct_sync, generated_metadata,
    image_response_value, insert_generated_file, lock_files, parse_image_json_request,
    previous_response_context, reject_unsupported_generation_options, request_conversation_id,
    request_echo, request_f32, request_include_reasoning, request_max_tokens, request_metadata,
    request_model_id, request_n, request_optional_string, request_reasoning_mode,
    request_response_format_instruction, request_stop_strings, request_store, request_suffix,
    request_u32, request_u64_opt, require_known_model, response_input_items,
    responses_prompt_for_reasoning, shared_fork_batch_supported, split_generated_reasoning,
    store_response_if_requested, unix_seconds, usage,
};

pub(crate) async fn create_batch(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let input_file_id = body
            .get("input_file_id")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::bad_request("batch requires input_file_id"))?
            .to_string();
        if !lock_files(&state)?.contains_key(&input_file_id) {
            return Err(ApiError::not_found(format!(
                "input file '{input_file_id}' does not exist"
            )));
        }
        let endpoint = body
            .get("endpoint")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::bad_request("batch requires endpoint"))?;
        let endpoint = normalize_batch_endpoint(endpoint)?;
        let completion_window = body
            .get("completion_window")
            .and_then(Value::as_str)
            .unwrap_or("24h")
            .to_string();
        let metadata = body.get("metadata").cloned().unwrap_or(Value::Null);
        let now = unix_seconds();
        let id = state.next_response_id("batch");
        let record = BatchRecord {
            id: id.clone(),
            object: "batch",
            endpoint,
            input_file_id,
            completion_window,
            status: "validating".to_string(),
            created_at: now,
            in_progress_at: None,
            finalizing_at: None,
            completed_at: None,
            failed_at: None,
            cancelled_at: None,
            expires_at: Some(now.saturating_add(24 * 60 * 60)),
            output_file_id: None,
            error_file_id: None,
            request_counts: BatchRequestCounts::default(),
            metadata,
            errors: Vec::new(),
        };
        lock_batches(&state)?.insert(id.clone(), record.clone());
        let state_for_job = state.clone();
        actix_web::rt::task::spawn_blocking(move || run_batch_job(state_for_job, id));
        Ok::<_, ApiError>(HttpResponse::Ok().json(batch_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_batches(state: web::Data<AppState>, request: HttpRequest) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let batches = lock_batches(&state)?
            .values()
            .map(batch_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "list",
            "data": batches
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_batch(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let batches = lock_batches(&state)?;
        let record = batches
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("batch '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(batch_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn cancel_batch(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let now = unix_seconds();
        let mut batches = lock_batches(&state)?;
        let record = batches
            .get_mut(&id)
            .ok_or_else(|| ApiError::not_found(format!("batch '{id}' does not exist")))?;
        if !matches!(
            record.status.as_str(),
            "completed" | "failed" | "expired" | "cancelled"
        ) {
            record.status = "cancelled".to_string();
            record.cancelled_at = Some(now);
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(batch_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn normalize_batch_endpoint(endpoint: &str) -> Result<String, ApiError> {
    let endpoint = endpoint.trim();
    let path = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint
            .find("/v1/")
            .map(|index| &endpoint[index..])
            .ok_or_else(|| ApiError::bad_request("batch endpoint URL must contain /v1/ path"))?
    } else {
        endpoint
    };
    match path {
        "/v1/completions"
        | "/v1/chat/completions"
        | "/v1/responses"
        | "/v1/embeddings"
        | "/v1/moderations"
        | "/v1/audio/transcriptions"
        | "/v1/audio/translations"
        | "/v1/audio/speech"
        | "/v1/images/generations"
        | "/v1/images/edits"
        | "/v1/images/variations"
        | "/v1/videos"
        | "/v1/videos/edits"
        | "/v1/videos/extensions" => Ok(path.to_string()),
        _ => Err(ApiError::unsupported(format!(
            "batch endpoint '{path}' is not implemented"
        ))),
    }
}

fn run_batch_job(state: web::Data<AppState>, batch_id: String) {
    if let Err(error) = run_batch_job_inner(&state, &batch_id) {
        mark_batch_failed(&state, &batch_id, error.message);
    }
}

fn run_batch_job_inner(state: &AppState, batch_id: &str) -> Result<(), ApiError> {
    let (endpoint, input_file_id) = {
        let mut batches = lock_batches(state)?;
        let batch = batches
            .get_mut(batch_id)
            .ok_or_else(|| ApiError::not_found(format!("batch '{batch_id}' does not exist")))?;
        batch.status = "in_progress".to_string();
        batch.in_progress_at = Some(unix_seconds());
        (batch.endpoint.clone(), batch.input_file_id.clone())
    };
    let input = {
        let files = lock_files(state)?;
        files
            .get(&input_file_id)
            .map(|file| file.content.clone())
            .ok_or_else(|| {
                ApiError::not_found(format!("input file '{input_file_id}' does not exist"))
            })?
    };
    let input = String::from_utf8(input).map_err(|err| {
        ApiError::bad_request(format!("batch input file must be UTF-8 JSONL: {err}"))
    })?;
    let mut output_lines = Vec::new();
    let mut error_lines = Vec::new();
    let mut counts = BatchRequestCounts::default();
    let mut errors = Vec::new();
    for (line_index, line) in input.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if batch_cancelled(state, batch_id) {
            return Ok(());
        }
        counts.total = counts.total.saturating_add(1);
        match run_batch_line(state, &endpoint, line, line_index) {
            Ok(line) => {
                output_lines.push(line.to_string());
                counts.completed = counts.completed.saturating_add(1);
            }
            Err(line) => {
                errors.push(batch_error_summary(&line));
                error_lines.push(line.to_string());
                counts.failed = counts.failed.saturating_add(1);
            }
        }
        update_batch_counts(state, batch_id, counts, errors.clone())?;
    }
    {
        let mut batches = lock_batches(state)?;
        if let Some(batch) = batches.get_mut(batch_id) {
            if batch.status == "cancelled" {
                return Ok(());
            }
            batch.status = "finalizing".to_string();
            batch.finalizing_at = Some(unix_seconds());
        }
    }
    let output_file_id = (!output_lines.is_empty()).then(|| {
        insert_generated_file(
            state,
            format!("{batch_id}_output.jsonl"),
            "batch_output",
            output_lines.join("\n").into_bytes(),
        )
    });
    let error_file_id = (!error_lines.is_empty()).then(|| {
        insert_generated_file(
            state,
            format!("{batch_id}_errors.jsonl"),
            "batch_error",
            error_lines.join("\n").into_bytes(),
        )
    });
    let mut batches = lock_batches(state)?;
    if let Some(batch) = batches.get_mut(batch_id) {
        if batch.status != "cancelled" {
            batch.status = "completed".to_string();
            batch.completed_at = Some(unix_seconds());
            batch.output_file_id = output_file_id.transpose()?;
            batch.error_file_id = error_file_id.transpose()?;
            batch.request_counts = counts;
            batch.errors = errors;
        }
    }
    Ok(())
}

fn run_batch_line(
    state: &AppState,
    batch_endpoint: &str,
    line: &str,
    line_index: usize,
) -> Result<Value, Value> {
    let id = state.next_response_id("batch_req");
    let item: Value = serde_json::from_str(line).map_err(|err| {
        batch_error_line(
            id.clone(),
            format!("line-{line_index}"),
            "invalid_json",
            format!("batch JSONL line is invalid: {err}"),
        )
    })?;
    let custom_id = item
        .get("custom_id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("line-{line_index}"));
    let method = item.get("method").and_then(Value::as_str).unwrap_or("POST");
    if method != "POST" {
        return Err(batch_error_line(
            id,
            custom_id,
            "invalid_method",
            "batch requests must use POST",
        ));
    }
    let item_endpoint = item
        .get("url")
        .and_then(Value::as_str)
        .map(normalize_batch_endpoint)
        .transpose()
        .map_err(|err| batch_error_line(id.clone(), custom_id.clone(), err.code, err.message))?
        .unwrap_or_else(|| batch_endpoint.to_string());
    if item_endpoint != batch_endpoint {
        return Err(batch_error_line(
            id,
            custom_id,
            "endpoint_mismatch",
            format!("batch item endpoint '{item_endpoint}' does not match '{batch_endpoint}'"),
        ));
    }
    let body = item.get("body").cloned().unwrap_or_else(|| json!({}));
    match execute_batch_generation_sync(state, &item_endpoint, &body) {
        Ok(body) => Ok(json!({
            "id": id,
            "custom_id": custom_id,
            "response": {
                "status_code": 200,
                "request_id": id,
                "body": body
            },
            "error": null
        })),
        Err(error) => Err(batch_error_line(id, custom_id, error.code, error.message)),
    }
}

fn execute_batch_generation_sync(
    state: &AppState,
    endpoint: &str,
    body: &Value,
) -> Result<Value, ApiError> {
    match endpoint {
        "/v1/embeddings" => create_embeddings_response(state, body),
        "/v1/moderations" => create_moderation_response(state, body),
        "/v1/audio/transcriptions" => create_audio_text_batch_response(body, false),
        "/v1/audio/translations" => create_audio_text_batch_response(body, true),
        "/v1/audio/speech" => create_speech_batch_response(body),
        "/v1/images/generations" => batch_image_response(body, ImageTask::Generation),
        "/v1/images/edits" => batch_image_response(body, ImageTask::Edit),
        "/v1/images/variations" => batch_image_response(body, ImageTask::Variation),
        "/v1/videos" => create_video_batch_response(state, VideoOperation::Create, body),
        "/v1/videos/edits" => create_video_batch_response(state, VideoOperation::Edit, body),
        "/v1/videos/extensions" => create_video_batch_response(state, VideoOperation::Extend, body),
        "/v1/completions" => {
            require_known_model(state, body)?;
            reject_unsupported_generation_options(body)?;
            batch_completion_response_sync(state, body)
        }
        "/v1/chat/completions" => {
            require_known_model(state, body)?;
            reject_unsupported_generation_options(body)?;
            batch_chat_completion_response_sync(state, body)
        }
        "/v1/responses" => {
            require_known_model(state, body)?;
            reject_unsupported_generation_options(body)?;
            batch_response_sync(state, body)
        }
        _ => Err(ApiError::unsupported(format!(
            "batch endpoint '{endpoint}' is not implemented"
        ))),
    }
}

fn batch_image_response(body: &Value, task: ImageTask) -> Result<Value, ApiError> {
    let parsed = parse_image_json_request(task, body)?;
    Ok(image_response_value(&parsed, unix_seconds()))
}

fn batch_completion_response_sync(state: &AppState, body: &Value) -> Result<Value, ApiError> {
    let response_model = request_model_id(state, body);
    let n = request_n(body)?;
    let prompts = completion_prompts(body)?;
    let max_tokens = request_max_tokens(state, body)?;
    let temperature = request_f32(body, "temperature", DEFAULT_TEMPERATURE)?;
    let top_p = request_f32(body, "top_p", DEFAULT_TOP_P)?;
    let top_k = request_u32(body, "top_k", 0)?;
    let seed = request_u64_opt(body, "seed")?;
    let stop = request_stop_strings(body)?;
    let session_id = request_optional_string(body, "session_id")?;
    let cache_key = request_optional_string(body, "cache_key")?;
    let echo = request_echo(body)?;
    let suffix = request_suffix(body)?;
    let response_format_instruction = request_response_format_instruction(body)?;
    let created = unix_seconds();
    let mut choices = Vec::new();
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    if n > 1 && prompts.len() == 1 && shared_fork_batch_supported(temperature, top_p, top_k, seed) {
        let prompt = prompts.into_iter().next().unwrap_or_else(empty_text_prompt);
        let output_prefix = completion_echo_prefix(&state.config.model_path, &prompt, echo)?;
        let prompt =
            apply_response_format_instruction(prompt, response_format_instruction.as_deref())?;
        let generated = generate_text_batch_direct_sync(
            state,
            GenerateOptions {
                prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                stop,
                session_id,
                cache_key,
                output_prefix,
                output_suffix: suffix.clone(),
                include_reasoning: false,
                reasoning_mode: ReasoningMode::None,
            },
            n,
        )?;
        for item in generated {
            prompt_tokens += item.prompt_tokens;
            completion_tokens += item.token_ids.len();
            choices.push(json!({
                "text": item.text,
                "index": choices.len(),
                "logprobs": null,
                "finish_reason": item.finish_reason,
                "nerva": generated_metadata(&item)
            }));
        }
    } else {
        for prompt in prompts {
            let output_prefix = completion_echo_prefix(&state.config.model_path, &prompt, echo)?;
            let prompt =
                apply_response_format_instruction(prompt, response_format_instruction.as_deref())?;
            for _ in 0..n {
                let generated = generate_text_direct_sync(
                    state,
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
                        output_prefix: output_prefix.clone(),
                        output_suffix: suffix.clone(),
                        include_reasoning: false,
                        reasoning_mode: ReasoningMode::None,
                    },
                )?;
                prompt_tokens += generated.prompt_tokens;
                completion_tokens += generated.token_ids.len();
                choices.push(json!({
                    "text": generated.text,
                    "index": choices.len(),
                    "logprobs": null,
                    "finish_reason": generated.finish_reason,
                    "nerva": generated_metadata(&generated)
                }));
            }
        }
    }
    Ok(json!({
        "id": state.next_response_id("cmpl"),
        "object": "text_completion",
        "created": created,
        "model": response_model,
        "choices": choices,
        "usage": usage(prompt_tokens, completion_tokens)
    }))
}

fn batch_chat_completion_response_sync(state: &AppState, body: &Value) -> Result<Value, ApiError> {
    let response_model = request_model_id(state, body);
    let n = request_n(body)?;
    let include_reasoning = request_include_reasoning(body)?;
    let reasoning_mode = request_reasoning_mode(state, body)?;
    let response_format_instruction = request_response_format_instruction(body)?;
    let prompt = apply_response_format_instruction(
        chat_prompt_for_reasoning(body, reasoning_mode)?,
        response_format_instruction.as_deref(),
    )?;
    let max_tokens = request_max_tokens(state, body)?;
    let temperature = request_f32(body, "temperature", DEFAULT_TEMPERATURE)?;
    let top_p = request_f32(body, "top_p", DEFAULT_TOP_P)?;
    let top_k = request_u32(body, "top_k", 0)?;
    let seed = request_u64_opt(body, "seed")?;
    let stop = request_stop_strings(body)?;
    let session_id = request_optional_string(body, "session_id")?;
    let cache_key = request_optional_string(body, "cache_key")?;
    let mut choices = Vec::with_capacity(n);
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    for index in 0..n {
        let generated = generate_text_direct_sync(
            state,
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
                output_prefix: None,
                output_suffix: None,
                include_reasoning,
                reasoning_mode,
            },
        )?;
        prompt_tokens += generated.prompt_tokens;
        completion_tokens += generated.token_ids.len();
        let split = split_generated_reasoning(&generated.text, reasoning_mode);
        let mut message = json!({
            "role": "assistant",
            "content": split.content
        });
        if include_reasoning && !split.reasoning.is_empty() {
            message["reasoning"] = json!(split.reasoning);
            message["reasoning_content"] = json!(message["reasoning"].as_str().unwrap_or(""));
        }
        choices.push(json!({
            "index": index,
            "message": message,
            "logprobs": null,
            "finish_reason": generated.finish_reason,
            "nerva": generated_metadata(&generated)
        }));
    }
    Ok(json!({
        "id": state.next_response_id("chatcmpl"),
        "object": "chat.completion",
        "created": unix_seconds(),
        "model": response_model,
        "choices": choices,
        "usage": usage(prompt_tokens, completion_tokens)
    }))
}

fn batch_response_sync(state: &AppState, body: &Value) -> Result<Value, ApiError> {
    let response_model = request_model_id(state, body);
    let store = request_store(body)?;
    let metadata = request_metadata(body)?;
    let previous_response_id = request_optional_string(body, "previous_response_id")?;
    let conversation_id = request_conversation_id(body)?;
    let conversation_context = conversation_context(state, conversation_id.as_deref())?;
    let previous_context = previous_response_context(state, previous_response_id.as_deref())?;
    let input_items = response_input_items(body);
    let include_reasoning = request_include_reasoning(body)?;
    let reasoning_mode = request_reasoning_mode(state, body)?;
    let response_format_instruction = request_response_format_instruction(body)?;
    let mut prompt = apply_response_format_instruction(
        responses_prompt_for_reasoning(body, reasoning_mode)?,
        response_format_instruction.as_deref(),
    )?;
    if let Some(previous_context) = previous_context {
        prompt = match prompt {
            PromptInput::Text { text, format } => PromptInput::Text {
                text: format!("{previous_context}{text}"),
                format,
            },
            prompt => prompt,
        };
    }
    if let Some(conversation_context) = conversation_context {
        prompt = match prompt {
            PromptInput::Text { text, format } => PromptInput::Text {
                text: format!("{conversation_context}{text}"),
                format,
            },
            prompt => prompt,
        };
    }
    let generated = generate_text_direct_sync(
        state,
        GenerateOptions {
            prompt,
            max_tokens: request_max_tokens(state, body)?,
            temperature: request_f32(body, "temperature", DEFAULT_TEMPERATURE)?,
            top_p: request_f32(body, "top_p", DEFAULT_TOP_P)?,
            top_k: request_u32(body, "top_k", 0)?,
            seed: request_u64_opt(body, "seed")?,
            stop: request_stop_strings(body)?,
            session_id: request_optional_string(body, "session_id")?,
            cache_key: request_optional_string(body, "cache_key")?,
            output_prefix: None,
            output_suffix: None,
            include_reasoning,
            reasoning_mode,
        },
    )?;
    let completion_tokens = generated.token_ids.len();
    let split = split_generated_reasoning(&generated.text, reasoning_mode);
    let mut output = Vec::new();
    if include_reasoning && !split.reasoning.is_empty() {
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
        "id": state.next_response_id("msg"),
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{
            "id": state.next_response_id("ct"),
            "type": "output_text",
            "text": split.content,
            "annotations": []
        }]
    }));
    let response_json = json!({
        "id": state.next_response_id("resp"),
        "object": "response",
        "created_at": unix_seconds(),
        "status": "completed",
        "error": null,
        "incomplete_details": null,
        "model": response_model,
        "metadata": metadata.clone(),
        "store": store,
        "previous_response_id": previous_response_id,
        "conversation": conversation_id.clone(),
        "output": output,
        "output_text": split.content,
        "nerva": generated_metadata(&generated),
        "usage": {
            "input_tokens": generated.prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": generated.prompt_tokens + completion_tokens
        }
    });
    let response_json =
        store_response_if_requested(state, response_json, input_items.clone(), store)?;
    append_response_to_conversation(
        state,
        conversation_id.as_deref(),
        &input_items,
        &response_json,
    )?;
    Ok(response_json)
}

fn update_batch_counts(
    state: &AppState,
    batch_id: &str,
    counts: BatchRequestCounts,
    errors: Vec<Value>,
) -> Result<(), ApiError> {
    let mut batches = lock_batches(state)?;
    if let Some(batch) = batches.get_mut(batch_id) {
        batch.request_counts = counts;
        batch.errors = errors;
    }
    Ok(())
}

fn batch_cancelled(state: &AppState, batch_id: &str) -> bool {
    lock_batches(state)
        .ok()
        .and_then(|batches| {
            batches
                .get(batch_id)
                .map(|batch| batch.status == "cancelled")
        })
        .unwrap_or(false)
}

fn mark_batch_failed(state: &AppState, batch_id: &str, message: String) {
    if let Ok(mut batches) = lock_batches(state) {
        if let Some(batch) = batches.get_mut(batch_id) {
            batch.status = "failed".to_string();
            batch.failed_at = Some(unix_seconds());
            batch.errors.push(json!({
                "code": "batch_failed",
                "message": message,
                "param": null,
                "line": null
            }));
        }
    }
}

fn batch_error_line(
    id: String,
    custom_id: String,
    code: impl Into<String>,
    message: impl Into<String>,
) -> Value {
    json!({
        "id": id,
        "custom_id": custom_id,
        "response": null,
        "error": {
            "code": code.into(),
            "message": message.into()
        }
    })
}

fn batch_error_summary(line: &Value) -> Value {
    json!({
        "code": line
            .get("error")
            .and_then(|error| error.get("code"))
            .and_then(Value::as_str)
            .unwrap_or("batch_item_failed"),
        "message": line
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("batch item failed"),
        "param": line.get("custom_id").cloned().unwrap_or(Value::Null),
        "line": null
    })
}

fn lock_batches(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, BatchRecord>>, ApiError> {
    state
        .batches
        .lock()
        .map_err(|_| ApiError::internal("batch registry lock poisoned"))
}

fn batch_json(record: &BatchRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "endpoint": record.endpoint,
        "input_file_id": record.input_file_id,
        "completion_window": record.completion_window,
        "status": record.status,
        "created_at": record.created_at,
        "in_progress_at": record.in_progress_at,
        "expires_at": record.expires_at,
        "finalizing_at": record.finalizing_at,
        "completed_at": record.completed_at,
        "failed_at": record.failed_at,
        "cancelled_at": record.cancelled_at,
        "output_file_id": record.output_file_id,
        "error_file_id": record.error_file_id,
        "request_counts": {
            "total": record.request_counts.total,
            "completed": record.request_counts.completed,
            "failed": record.request_counts.failed
        },
        "metadata": record.metadata,
        "errors": if record.errors.is_empty() {
            Value::Null
        } else {
            json!({
                "object": "list",
                "data": record.errors.clone()
            })
        }
    })
}
