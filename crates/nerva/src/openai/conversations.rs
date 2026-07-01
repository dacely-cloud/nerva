use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Map, Value, json};

use super::{
    ApiError, AppState, ConversationRecord, authorize, message_content_text, request_metadata,
    unix_seconds,
};

pub(crate) async fn create_conversation(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let metadata = request_metadata(&body)?;
        let items = request_conversation_items(&body)?;
        let now = unix_seconds();
        let record = ConversationRecord {
            id: state.next_response_id("conv"),
            created_at: now,
            updated_at: now,
            metadata,
            items,
        };
        lock_conversations(&state)?.insert(record.id.clone(), record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(conversation_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_conversation(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let conversations = lock_conversations(&state)?;
        let record = conversations
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("conversation '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(conversation_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn update_conversation(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let metadata = request_metadata(&body)?;
        let mut conversations = lock_conversations(&state)?;
        let record = conversations
            .get_mut(&id)
            .ok_or_else(|| ApiError::not_found(format!("conversation '{id}' does not exist")))?;
        record.metadata = metadata;
        record.updated_at = unix_seconds();
        Ok::<_, ApiError>(HttpResponse::Ok().json(conversation_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_conversation(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let removed = lock_conversations(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "conversation.deleted",
            "deleted": removed
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_conversation_items(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let items = request_conversation_items(&body)?;
        let mut conversations = lock_conversations(&state)?;
        let record = conversations
            .get_mut(&id)
            .ok_or_else(|| ApiError::not_found(format!("conversation '{id}' does not exist")))?;
        record.items.extend(items);
        record.updated_at = unix_seconds();
        Ok::<_, ApiError>(HttpResponse::Ok().json(conversation_items_list_json(&record.items)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_conversation_items(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let conversations = lock_conversations(&state)?;
        let record = conversations
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("conversation '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(conversation_items_list_json(&record.items)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_conversation_item(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (conversation_id, item_id) = path.into_inner();
        let conversations = lock_conversations(&state)?;
        let record = conversations.get(&conversation_id).ok_or_else(|| {
            ApiError::not_found(format!("conversation '{conversation_id}' does not exist"))
        })?;
        let item = record
            .items
            .iter()
            .find(|item| item_id_of(item) == Some(item_id.as_str()));
        let item = item.ok_or_else(|| {
            ApiError::not_found(format!("conversation item '{item_id}' does not exist"))
        })?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(item.clone()))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_conversation_item(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (conversation_id, item_id) = path.into_inner();
        let mut conversations = lock_conversations(&state)?;
        let record = conversations.get_mut(&conversation_id).ok_or_else(|| {
            ApiError::not_found(format!("conversation '{conversation_id}' does not exist"))
        })?;
        let before = record.items.len();
        record
            .items
            .retain(|item| item_id_of(item) != Some(item_id.as_str()));
        let removed = record.items.len() != before;
        if removed {
            record.updated_at = unix_seconds();
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": item_id,
            "object": "conversation.item.deleted",
            "deleted": removed
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn request_conversation_id(body: &Value) -> Result<Option<String>, ApiError> {
    match body.get("conversation") {
        Some(Value::String(id)) if !id.trim().is_empty() => Ok(Some(id.clone())),
        Some(Value::Object(object)) => match object.get("id") {
            Some(Value::String(id)) if !id.trim().is_empty() => Ok(Some(id.clone())),
            Some(Value::String(_)) => {
                Err(ApiError::bad_request("conversation id must not be empty"))
            }
            Some(_) => Err(ApiError::bad_request("conversation id must be a string")),
            None => Err(ApiError::bad_request("conversation object requires id")),
        },
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(
            "conversation must be a string id or an object with id",
        )),
    }
}

pub(crate) fn conversation_context(
    state: &AppState,
    conversation_id: Option<&str>,
) -> Result<Option<String>, ApiError> {
    let Some(conversation_id) = conversation_id else {
        return Ok(None);
    };
    let conversations = lock_conversations(state)?;
    let record = conversations.get(conversation_id).ok_or_else(|| {
        ApiError::not_found(format!("conversation '{conversation_id}' does not exist"))
    })?;
    let text = conversation_items_prompt(&record.items)?;
    if text.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(text))
    }
}

pub(crate) fn append_response_to_conversation(
    state: &AppState,
    conversation_id: Option<&str>,
    input_items: &[Value],
    response: &Value,
) -> Result<(), ApiError> {
    let Some(conversation_id) = conversation_id else {
        return Ok(());
    };
    let output_items = response
        .get("output")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut conversations = lock_conversations(state)?;
    let record = conversations.get_mut(conversation_id).ok_or_else(|| {
        ApiError::not_found(format!("conversation '{conversation_id}' does not exist"))
    })?;
    record
        .items
        .extend(input_items.iter().cloned().map(normalize_conversation_item));
    record
        .items
        .extend(output_items.into_iter().map(normalize_conversation_item));
    record.updated_at = unix_seconds();
    Ok(())
}

pub(crate) fn conversation_items_prompt(items: &[Value]) -> Result<String, ApiError> {
    let mut prompt = String::new();
    for item in items {
        let item_type = item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        if item_type != "message" {
            continue;
        }
        let content = message_content_text(item.get("content"))?;
        if content.trim().is_empty() {
            continue;
        }
        match item.get("role").and_then(Value::as_str).unwrap_or("user") {
            "system" | "developer" => prompt.push_str("System: "),
            "assistant" => prompt.push_str("Assistant: "),
            "tool" => prompt.push_str("Tool: "),
            _ => prompt.push_str("User: "),
        }
        prompt.push_str(&content);
        prompt.push('\n');
    }
    Ok(prompt)
}

fn request_conversation_items(body: &Value) -> Result<Vec<Value>, ApiError> {
    match (body.get("items"), body.get("item")) {
        (Some(Value::Array(items)), _) => Ok(items
            .iter()
            .cloned()
            .map(normalize_conversation_item)
            .collect()),
        (Some(Value::Object(_)), _) => Ok(vec![normalize_conversation_item(body["items"].clone())]),
        (Some(Value::Null), _) | (None, Some(Value::Null)) | (None, None) => Ok(Vec::new()),
        (None, Some(Value::Object(_))) => {
            Ok(vec![normalize_conversation_item(body["item"].clone())])
        }
        (None, Some(Value::Array(items))) => Ok(items
            .iter()
            .cloned()
            .map(normalize_conversation_item)
            .collect()),
        (Some(_), _) | (None, Some(_)) => Err(ApiError::bad_request(
            "conversation items must be an object or array",
        )),
    }
}

fn normalize_conversation_item(mut item: Value) -> Value {
    let stable_id = format!("item-{:016x}", stable_hash_value(&item));
    let Some(object) = item.as_object_mut() else {
        let mut normalized = Map::new();
        normalized.insert("id".to_string(), json!(stable_id));
        normalized.insert("type".to_string(), json!("message"));
        normalized.insert("role".to_string(), json!("user"));
        normalized.insert(
            "content".to_string(),
            json!([{"type": "input_text", "text": value_text(&item)}]),
        );
        return Value::Object(normalized);
    };
    if !object.contains_key("id") {
        object.insert("id".to_string(), json!(stable_id));
    }
    if !object.contains_key("type") {
        object.insert("type".to_string(), json!("message"));
    }
    if object.get("type").and_then(Value::as_str) == Some("message")
        && !object.contains_key("status")
    {
        object.insert("status".to_string(), json!("completed"));
    }
    item
}

fn conversation_json(record: &ConversationRecord) -> Value {
    json!({
        "id": record.id,
        "object": "conversation",
        "created_at": record.created_at,
        "metadata": record.metadata
    })
}

fn conversation_items_list_json(items: &[Value]) -> Value {
    json!({
        "object": "list",
        "data": items,
        "first_id": items.first().and_then(item_id_value),
        "last_id": items.last().and_then(item_id_value),
        "has_more": false
    })
}

fn item_id_of(value: &Value) -> Option<&str> {
    value.get("id").and_then(Value::as_str)
}

fn item_id_value(value: &Value) -> Option<Value> {
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

fn lock_conversations(
    state: &AppState,
) -> Result<
    std::sync::MutexGuard<'_, std::collections::HashMap<String, ConversationRecord>>,
    ApiError,
> {
    state
        .conversations
        .lock()
        .map_err(|_| ApiError::internal("conversation registry lock poisoned"))
}
