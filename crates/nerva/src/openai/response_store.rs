use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{ApiError, AppState, StoredResponseRecord, authorize};

pub(crate) async fn get_response(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let responses = lock_responses(&state)?;
        let record = responses
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("response '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(record.response.clone()))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_response(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let removed = lock_responses(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "response.deleted",
            "deleted": removed
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_response_input_items(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let responses = lock_responses(&state)?;
        let record = responses
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("response '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "list",
            "data": record.input_items.clone(),
            "first_id": record.input_items.first().and_then(input_item_id),
            "last_id": record.input_items.last().and_then(input_item_id),
            "has_more": false
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn store_response_if_requested(
    state: &AppState,
    response: Value,
    input_items: Vec<Value>,
    store: bool,
) -> Result<Value, ApiError> {
    if !store {
        return Ok(response);
    }
    let id = response
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::internal("stored response is missing id"))?
        .to_string();
    lock_responses(state)?.insert(
        id,
        StoredResponseRecord {
            response: response.clone(),
            input_items,
        },
    );
    Ok(response)
}

pub(crate) fn previous_response_context(
    state: &AppState,
    previous_response_id: Option<&str>,
) -> Result<Option<String>, ApiError> {
    let Some(previous_response_id) = previous_response_id else {
        return Ok(None);
    };
    let responses = lock_responses(state)?;
    let record = responses.get(previous_response_id).ok_or_else(|| {
        ApiError::not_found(format!(
            "previous response '{previous_response_id}' does not exist"
        ))
    })?;
    let text = response_output_text(&record.response);
    if text.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(format!("Previous response output:\n{text}\n")))
    }
}

pub(crate) fn response_input_items(body: &Value) -> Vec<Value> {
    match body.get("input") {
        Some(Value::Array(items)) => items.clone(),
        Some(input) => vec![json!({
            "id": format!("input-{:x}", stable_hash_value(input)),
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": value_text(input)
            }]
        })],
        None => Vec::new(),
    }
}

pub(crate) fn response_output_text(response: &Value) -> String {
    if let Some(text) = response.get("output_text").and_then(Value::as_str) {
        return text.to_string();
    }
    let Some(output) = response.get("output").and_then(Value::as_array) else {
        return String::new();
    };
    let mut parts = Vec::new();
    for item in output {
        if item.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let Some(content) = item.get("content").and_then(Value::as_array) else {
            continue;
        };
        for part in content {
            if matches!(
                part.get("type").and_then(Value::as_str),
                Some("output_text") | Some("text") | None
            ) {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    parts.push(text.to_string());
                }
            }
        }
    }
    parts.join("\n")
}

fn lock_responses(
    state: &AppState,
) -> Result<
    std::sync::MutexGuard<'_, std::collections::HashMap<String, StoredResponseRecord>>,
    ApiError,
> {
    state
        .responses
        .lock()
        .map_err(|_| ApiError::internal("response store lock poisoned"))
}

fn input_item_id(value: &Value) -> Option<Value> {
    value.get("id").cloned()
}

fn value_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn stable_hash_value(value: &Value) -> u64 {
    let text = value.to_string();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
