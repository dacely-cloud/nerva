use std::collections::HashMap;
use std::sync::MutexGuard;
use std::sync::atomic::Ordering;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, ChatKitSessionRecord, ChatKitThreadItemRecord, ChatKitThreadRecord,
    authorize, percent_decode_query, request_metadata, unix_seconds,
};

const DEFAULT_CHATKIT_TTL_SECONDS: u64 = 600;
const DEFAULT_CHATKIT_RATE_LIMIT: u64 = 10;
const DEFAULT_CHATKIT_MAX_FILES: u64 = 10;
const DEFAULT_CHATKIT_MAX_FILE_SIZE_MB: u64 = 512;
const DEFAULT_CHATKIT_LIST_LIMIT: usize = 20;
const MAX_CHATKIT_LIST_LIMIT: usize = 100;

pub(crate) async fn create_chatkit_session(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        let now = unix_seconds();
        let id = state.next_response_id("cksess");
        let ttl_seconds = request_chatkit_ttl(&body)?;
        let thread = create_session_thread_if_requested(&state, &body)?;
        let record = ChatKitSessionRecord {
            id: id.clone(),
            object: "chatkit.session",
            created_at: now,
            expires_at: now.saturating_add(ttl_seconds),
            cancelled_at: None,
            status: "active".to_string(),
            client_secret: chatkit_client_secret(&id, now),
            workflow: request_chatkit_workflow(&body)?,
            scope: object_or_empty(&body, "scope")?,
            user: optional_string(&body, "user")?,
            chatkit_configuration: request_chatkit_configuration(&body)?,
            rate_limits: request_chatkit_rate_limits(&body)?,
            max_requests_per_1_minute: request_chatkit_rate_limit(&body)?,
            max_requests_per_session: optional_u64_alias(
                &body,
                &["max_requests_per_session", "max_requests"],
            )?,
            ttl_seconds,
            thread_id: thread.map(|thread| thread.id),
        };
        lock_chatkit_sessions(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(chatkit_session_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn cancel_chatkit_session(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let mut sessions = lock_chatkit_sessions(&state)?;
        let record = sessions
            .get_mut(&id)
            .ok_or_else(|| ApiError::not_found(format!("ChatKit session '{id}' does not exist")))?;
        if record.status == "active" {
            record.status = "cancelled".to_string();
            record.cancelled_at = Some(unix_seconds());
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(chatkit_session_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_chatkit_threads(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let query = ChatKitListQuery::from_request(&request)?;
        let threads = lock_chatkit_threads(&state)?;
        let mut records = threads
            .values()
            .filter(|thread| {
                query
                    .user
                    .as_deref()
                    .is_none_or(|user| thread.user.as_deref() == Some(user))
            })
            .collect::<Vec<_>>();
        records.sort_by_key(|thread| thread.created_at);
        if query.order != "asc" {
            records.reverse();
        }
        let records = page_records(records, &query, |thread| &thread.id);
        Ok::<_, ApiError>(HttpResponse::Ok().json(chatkit_list_json(records, chatkit_thread_json)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_chatkit_thread(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let threads = lock_chatkit_threads(&state)?;
        let record = threads
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("ChatKit thread '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(chatkit_thread_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_chatkit_thread(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let deleted = lock_chatkit_threads(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "chatkit.thread.deleted",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_chatkit_thread_items(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let thread_id = path.into_inner();
        let query = ChatKitListQuery::from_request(&request)?;
        let threads = lock_chatkit_threads(&state)?;
        let thread = threads.get(&thread_id).ok_or_else(|| {
            ApiError::not_found(format!("ChatKit thread '{thread_id}' does not exist"))
        })?;
        let mut records = thread.items.iter().collect::<Vec<_>>();
        if query.order != "asc" {
            records.reverse();
        }
        let records = page_records(records, &query, |item| &item.id);
        Ok::<_, ApiError>(
            HttpResponse::Ok().json(chatkit_list_json(records, chatkit_thread_item_json)),
        )
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

fn create_session_thread_if_requested(
    state: &AppState,
    body: &Value,
) -> Result<Option<ChatKitThreadRecord>, ApiError> {
    let requested = body.get("thread").is_some()
        || body.get("items").is_some()
        || body.get("initial_items").is_some()
        || body.get("message").is_some();
    if !requested {
        return Ok(None);
    }
    let thread_body = body.get("thread").unwrap_or(body);
    let record = chatkit_thread_from_body(state, thread_body, body)?;
    lock_chatkit_threads(state)?.insert(record.id.clone(), record.clone());
    Ok(Some(record))
}

fn chatkit_thread_from_body(
    state: &AppState,
    thread_body: &Value,
    session_body: &Value,
) -> Result<ChatKitThreadRecord, ApiError> {
    let now = unix_seconds();
    let user = optional_string(thread_body, "user")?.or(optional_string(session_body, "user")?);
    let title = optional_string(thread_body, "title")?;
    let items_source = thread_body
        .get("items")
        .or_else(|| thread_body.get("initial_items"))
        .or_else(|| session_body.get("items"))
        .or_else(|| session_body.get("initial_items"));
    let mut items = match items_source {
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| chatkit_thread_item_from_value(state, item))
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => {
            return Err(ApiError::bad_request(
                "ChatKit thread items must be an array",
            ));
        }
        None => Vec::new(),
    };
    if let Some(message) = session_body.get("message") {
        items.push(chatkit_thread_item_from_value(
            state,
            &json!({
                "type": "user_message",
                "content": message
            }),
        )?);
    }
    Ok(ChatKitThreadRecord {
        id: state.next_response_id("cthr"),
        object: "chatkit.thread",
        created_at: now,
        title: title.or_else(|| chatkit_title_from_items(&items)),
        user,
        status: status_or_active(thread_body)?,
        items,
        metadata: metadata_or_empty(thread_body)?,
    })
}

fn chatkit_thread_item_from_value(
    state: &AppState,
    value: &Value,
) -> Result<ChatKitThreadItemRecord, ApiError> {
    let object = value.as_object();
    let item_type = object
        .and_then(|object| object.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("user_message")
        .to_string();
    if !matches!(
        item_type.as_str(),
        "user_message"
            | "assistant_message"
            | "widget"
            | "tool_call"
            | "tool_result"
            | "system_message"
    ) {
        return Err(ApiError::unsupported(format!(
            "ChatKit thread item type '{item_type}' is not implemented"
        )));
    }
    let content = object
        .and_then(|object| object.get("content"))
        .unwrap_or(value);
    Ok(ChatKitThreadItemRecord {
        id: object
            .and_then(|object| object.get("id"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| state.next_response_id("cthi")),
        object: "chatkit.thread_item",
        created_at: object
            .and_then(|object| object.get("created_at"))
            .and_then(Value::as_u64)
            .unwrap_or_else(unix_seconds),
        item_type: item_type.clone(),
        content: normalize_chatkit_content(content, &item_type)?,
        attachments: object
            .and_then(|object| object.get("attachments"))
            .map(array_value)
            .transpose()?
            .unwrap_or_default(),
        metadata: object
            .and_then(|object| object.get("metadata"))
            .map(|metadata| {
                if metadata.is_object() {
                    Ok(metadata.clone())
                } else {
                    Err(ApiError::bad_request(
                        "ChatKit thread item metadata must be an object",
                    ))
                }
            })
            .transpose()?
            .unwrap_or_else(|| json!({})),
    })
}

pub(crate) fn normalize_chatkit_content(
    content: &Value,
    item_type: &str,
) -> Result<Vec<Value>, ApiError> {
    match content {
        Value::String(text) => Ok(vec![chatkit_text_part(text, item_type)]),
        Value::Array(parts) => parts
            .iter()
            .map(|part| normalize_chatkit_part(part, item_type))
            .collect(),
        Value::Object(_) => Ok(vec![normalize_chatkit_part(content, item_type)?]),
        _ => Err(ApiError::bad_request(
            "ChatKit content must be a string, object, or array",
        )),
    }
}

fn normalize_chatkit_part(part: &Value, item_type: &str) -> Result<Value, ApiError> {
    let Some(object) = part.as_object() else {
        return Err(ApiError::bad_request(
            "ChatKit content parts must be objects",
        ));
    };
    let part_type = object.get("type").and_then(Value::as_str).unwrap_or("");
    if matches!(
        part_type,
        "input_text"
            | "output_text"
            | "text"
            | "input_image"
            | "output_image"
            | "file"
            | "refusal"
            | "widget"
    ) {
        let mut normalized = object.clone();
        if part_type == "text" {
            normalized.insert(
                "type".to_string(),
                Value::String(default_text_part_type(item_type).to_string()),
            );
        }
        if !normalized.contains_key("text")
            && let Some(value) = normalized.get("value").and_then(Value::as_str)
        {
            normalized.insert("text".to_string(), Value::String(value.to_string()));
        }
        Ok(Value::Object(normalized))
    } else {
        Err(ApiError::unsupported(format!(
            "ChatKit content part '{part_type}' is not implemented"
        )))
    }
}

fn chatkit_text_part(text: &str, item_type: &str) -> Value {
    json!({
        "type": default_text_part_type(item_type),
        "text": text
    })
}

fn default_text_part_type(item_type: &str) -> &'static str {
    if item_type == "assistant_message" {
        "output_text"
    } else {
        "input_text"
    }
}

pub(crate) fn chatkit_session_json(record: &ChatKitSessionRecord) -> Value {
    let mut value = json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "workflow": record.workflow,
        "scope": record.scope,
        "chatkit_configuration": record.chatkit_configuration,
        "client_secret": record.client_secret,
        "expires_at": record.expires_at,
        "status": record.status,
        "user": record.user,
        "rate_limits": record.rate_limits,
        "max_requests_per_1_minute": record.max_requests_per_1_minute,
        "max_requests_per_session": record.max_requests_per_session,
        "ttl_seconds": record.ttl_seconds,
        "cancelled_at": record.cancelled_at
    });
    if let Some(thread_id) = record.thread_id.as_deref() {
        value["thread_id"] = json!(thread_id);
        value["nerva"] = json!({"thread_id": thread_id});
    }
    value
}

pub(crate) fn chatkit_thread_json(record: &ChatKitThreadRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "title": record.title,
        "user": record.user,
        "status": record.status,
        "metadata": record.metadata,
        "items": chatkit_list_json(record.items.iter().collect::<Vec<_>>(), chatkit_thread_item_json)
    })
}

pub(crate) fn chatkit_thread_item_json(record: &ChatKitThreadItemRecord) -> Value {
    let mut value = json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "type": record.item_type,
        "content": record.content,
        "attachments": record.attachments
    });
    if record.metadata != json!({}) {
        value["metadata"] = record.metadata.clone();
    }
    value
}

fn chatkit_list_json<T>(records: Vec<&T>, render: fn(&T) -> Value) -> Value {
    let data = records
        .iter()
        .map(|record| render(record))
        .collect::<Vec<_>>();
    json!({
        "object": "list",
        "data": data,
        "first_id": records.first().and_then(|record| record_id(render(record))),
        "last_id": records.last().and_then(|record| record_id(render(record))),
        "has_more": false
    })
}

fn record_id(value: Value) -> Option<String> {
    value.get("id").and_then(Value::as_str).map(str::to_string)
}

struct ChatKitListQuery {
    order: String,
    limit: usize,
    after: Option<String>,
    before: Option<String>,
    user: Option<String>,
}

impl ChatKitListQuery {
    fn from_request(request: &HttpRequest) -> Result<Self, ApiError> {
        let query = request.query_string();
        let order = query_param(query, "order").unwrap_or_else(|| "desc".to_string());
        if !matches!(order.as_str(), "asc" | "desc") {
            return Err(ApiError::bad_request("order must be asc or desc"));
        }
        let limit = query_param(query, "limit")
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| ApiError::bad_request("limit must be a number"))
            })
            .transpose()?
            .unwrap_or(DEFAULT_CHATKIT_LIST_LIMIT)
            .min(MAX_CHATKIT_LIST_LIMIT);
        Ok(Self {
            order,
            limit,
            after: query_param(query, "after"),
            before: query_param(query, "before"),
            user: query_param(query, "user"),
        })
    }
}

fn page_records<'a, T>(
    records: Vec<&'a T>,
    query: &ChatKitListQuery,
    id: fn(&T) -> &str,
) -> Vec<&'a T> {
    records
        .into_iter()
        .skip_while(|record| {
            query
                .after
                .as_deref()
                .is_some_and(|after| id(record) != after)
        })
        .skip(usize::from(query.after.is_some()))
        .take_while(|record| {
            query
                .before
                .as_deref()
                .is_none_or(|before| id(record) != before)
        })
        .take(query.limit)
        .collect()
}

fn request_chatkit_workflow(body: &Value) -> Result<Value, ApiError> {
    let mut workflow = object_or_default(body, "workflow", || json!({"id": "nerva"}))?;
    if workflow.get("id").and_then(Value::as_str).is_none() {
        workflow["id"] = json!("nerva");
    }
    if workflow.get("tracing").is_none() {
        workflow["tracing"] = json!({"enabled": true});
    }
    if workflow.get("state_variables").is_none() {
        workflow["state_variables"] = Value::Null;
    }
    if workflow.get("version").is_none() {
        workflow["version"] = Value::Null;
    }
    Ok(workflow)
}

fn request_chatkit_configuration(body: &Value) -> Result<Value, ApiError> {
    let mut config = object_or_default(body, "chatkit_configuration", || json!({}))?;
    let automatic_thread_titling = config
        .get("automatic_thread_titling")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let file_upload = config
        .get("file_upload")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let history = config.get("history").cloned().unwrap_or_else(|| json!({}));
    config["automatic_thread_titling"] = json!({
        "enabled": optional_bool(&automatic_thread_titling, "enabled")?.unwrap_or(true)
    });
    config["file_upload"] = json!({
        "enabled": optional_bool(&file_upload, "enabled")?.unwrap_or(false),
        "max_file_size": optional_u64(&file_upload, "max_file_size")?
            .unwrap_or(DEFAULT_CHATKIT_MAX_FILE_SIZE_MB),
        "max_files": optional_u64(&file_upload, "max_files")?
            .unwrap_or(DEFAULT_CHATKIT_MAX_FILES)
    });
    config["history"] = json!({
        "enabled": optional_bool(&history, "enabled")?.unwrap_or(true),
        "recent_threads": optional_u64(&history, "recent_threads")?
    });
    Ok(config)
}

fn request_chatkit_rate_limits(body: &Value) -> Result<Value, ApiError> {
    let rate_limits = object_or_empty(body, "rate_limits")?;
    let max = optional_u64(&rate_limits, "max_requests_per_1_minute")?
        .or(optional_u64(body, "max_requests_per_1_minute")?)
        .unwrap_or(DEFAULT_CHATKIT_RATE_LIMIT);
    Ok(json!({ "max_requests_per_1_minute": max }))
}

fn request_chatkit_rate_limit(body: &Value) -> Result<Option<u64>, ApiError> {
    Ok(Some(
        optional_u64(body, "max_requests_per_1_minute")?
            .or_else(|| {
                body.get("rate_limits")
                    .and_then(|value| value.get("max_requests_per_1_minute"))
                    .and_then(Value::as_u64)
            })
            .unwrap_or(DEFAULT_CHATKIT_RATE_LIMIT),
    ))
}

fn request_chatkit_ttl(body: &Value) -> Result<u64, ApiError> {
    match body.get("expires_after") {
        Some(Value::Number(number)) => number
            .as_u64()
            .filter(|value| *value > 0)
            .ok_or_else(|| ApiError::bad_request("expires_after must be positive")),
        Some(Value::Object(object)) => object
            .get("seconds")
            .and_then(Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| ApiError::bad_request("expires_after.seconds must be positive")),
        Some(Value::Null) | None => Ok(DEFAULT_CHATKIT_TTL_SECONDS),
        Some(_) => Err(ApiError::bad_request(
            "expires_after must be a number or object",
        )),
    }
}

fn status_or_active(body: &Value) -> Result<Value, ApiError> {
    match body.get("status") {
        Some(Value::Object(status)) => {
            let ty = status
                .get("type")
                .and_then(Value::as_str)
                .ok_or_else(|| ApiError::bad_request("ChatKit thread status requires type"))?;
            if !matches!(ty, "active" | "locked" | "closed") {
                return Err(ApiError::bad_request(
                    "ChatKit thread status type must be active, locked, or closed",
                ));
            }
            Ok(Value::Object(status.clone()))
        }
        Some(Value::String(ty)) if matches!(ty.as_str(), "active" | "locked" | "closed") => {
            Ok(json!({"type": ty}))
        }
        Some(Value::Null) | None => Ok(json!({"type": "active"})),
        Some(_) => Err(ApiError::bad_request(
            "ChatKit thread status must be an object or string",
        )),
    }
}

fn chatkit_title_from_items(items: &[ChatKitThreadItemRecord]) -> Option<String> {
    items
        .iter()
        .find(|item| item.item_type == "user_message")
        .and_then(|item| {
            item.content
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .find(|text| !text.trim().is_empty())
        })
        .map(|text| text.chars().take(80).collect())
}

fn optional_string(body: &Value, field: &'static str) -> Result<Option<String>, ApiError> {
    match body.get(field) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.clone())),
        Some(Value::String(_)) => Err(ApiError::bad_request(format!("{field} must not be empty"))),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be a string"))),
    }
}

fn optional_u64(body: &Value, field: &'static str) -> Result<Option<u64>, ApiError> {
    match body.get(field) {
        Some(Value::Number(number)) => number
            .as_u64()
            .map(Some)
            .ok_or_else(|| ApiError::bad_request(format!("{field} must be a positive integer"))),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be a number"))),
    }
}

fn optional_u64_alias(body: &Value, fields: &[&'static str]) -> Result<Option<u64>, ApiError> {
    for field in fields {
        if body.get(*field).is_some() {
            return optional_u64(body, field);
        }
    }
    Ok(None)
}

fn optional_bool(body: &Value, field: &'static str) -> Result<Option<bool>, ApiError> {
    match body.get(field) {
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be a boolean"))),
    }
}

fn array_value(value: &Value) -> Result<Vec<Value>, ApiError> {
    match value {
        Value::Array(items) => Ok(items.clone()),
        Value::Null => Ok(Vec::new()),
        _ => Err(ApiError::bad_request("value must be an array")),
    }
}

fn object_or_empty(body: &Value, field: &'static str) -> Result<Value, ApiError> {
    object_or_default(body, field, || json!({}))
}

fn object_or_default(
    body: &Value,
    field: &'static str,
    default: impl FnOnce() -> Value,
) -> Result<Value, ApiError> {
    match body.get(field) {
        Some(Value::Object(object)) => Ok(Value::Object(object.clone())),
        Some(Value::Null) | None => Ok(default()),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be an object"))),
    }
}

fn metadata_or_empty(body: &Value) -> Result<Value, ApiError> {
    match request_metadata(body)? {
        Value::Null => Ok(json!({})),
        value => Ok(value),
    }
}

fn query_param(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|part| {
        let (key, value) = part.split_once('=')?;
        (percent_decode_query(key) == name).then(|| percent_decode_query(value))
    })
}

fn chatkit_client_secret(session_id: &str, created_at: u64) -> String {
    let mut hash = created_at ^ 0x8df6_0f64_719d_77a5;
    for byte in session_id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    format!("chatkit_token_{hash:016x}")
}

fn lock_chatkit_sessions(
    state: &AppState,
) -> Result<MutexGuard<'_, HashMap<String, ChatKitSessionRecord>>, ApiError> {
    state
        .chatkit_sessions
        .lock()
        .map_err(|_| ApiError::internal("ChatKit session registry lock poisoned"))
}

fn lock_chatkit_threads(
    state: &AppState,
) -> Result<MutexGuard<'_, HashMap<String, ChatKitThreadRecord>>, ApiError> {
    state
        .chatkit_threads
        .lock()
        .map_err(|_| ApiError::internal("ChatKit thread registry lock poisoned"))
}
