use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, StoredChatCompletionRecord, authorize, request_metadata, unix_seconds,
};

pub(crate) async fn list_chat_completions(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let data = lock_chat_completions(&state)?
            .values()
            .map(|record| record.response.clone())
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "list",
            "data": data,
            "first_id": data.first().and_then(value_id),
            "last_id": data.last().and_then(value_id),
            "has_more": false
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_chat_completion(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let completions = lock_chat_completions(&state)?;
        let record = completions
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("chat completion '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(record.response.clone()))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn update_chat_completion(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let metadata = request_metadata(&body)?;
        let mut completions = lock_chat_completions(&state)?;
        let record = completions
            .get_mut(&id)
            .ok_or_else(|| ApiError::not_found(format!("chat completion '{id}' does not exist")))?;
        record.metadata = metadata.clone();
        record.response["metadata"] = metadata;
        record.response["updated"] = json!(unix_seconds());
        Ok::<_, ApiError>(HttpResponse::Ok().json(record.response.clone()))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_chat_completion(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let removed = lock_chat_completions(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "chat.completion.deleted",
            "deleted": removed
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_chat_completion_messages(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let completions = lock_chat_completions(&state)?;
        let record = completions
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("chat completion '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "list",
            "data": record.messages,
            "first_id": record.messages.first().and_then(value_id),
            "last_id": record.messages.last().and_then(value_id),
            "has_more": false
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn store_chat_completion_if_requested(
    state: &AppState,
    response: Value,
    request_messages: Vec<Value>,
    metadata: Value,
    store: bool,
) -> Result<Value, ApiError> {
    if !store {
        return Ok(response);
    }
    let id = response
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::internal("stored chat completion is missing id"))?
        .to_string();
    let mut messages = request_messages;
    for choice in response
        .get("choices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(message) = choice.get("message") {
            messages.push(message.clone());
        }
    }
    lock_chat_completions(state)?.insert(
        id,
        StoredChatCompletionRecord {
            response: response.clone(),
            messages,
            metadata,
        },
    );
    Ok(response)
}

pub(crate) fn chat_completion_request_messages(body: &Value) -> Vec<Value> {
    body.get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn lock_chat_completions(
    state: &AppState,
) -> Result<
    std::sync::MutexGuard<'_, std::collections::HashMap<String, StoredChatCompletionRecord>>,
    ApiError,
> {
    state
        .chat_completions
        .lock()
        .map_err(|_| ApiError::internal("chat completion store lock poisoned"))
}

fn value_id(value: &Value) -> Option<Value> {
    value.get("id").cloned()
}
