use std::collections::HashMap;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, VideoCharacterRecord, VideoRecord, authorize, request_metadata,
    unix_seconds,
};

const DEFAULT_VIDEO_MODEL: &str = "sora-2";
const DEFAULT_VIDEO_SECONDS: &str = "4";
const DEFAULT_VIDEO_SIZE: &str = "720x1280";
const VIDEO_ASSET_TTL_SECONDS: u64 = 60 * 60;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum VideoOperation {
    Create,
    Edit,
    Extend,
    Remix,
}

#[derive(Clone, Debug)]
pub(crate) struct ParsedVideoRequest {
    pub(crate) operation: VideoOperation,
    pub(crate) model: String,
    pub(crate) prompt: String,
    pub(crate) seconds: String,
    pub(crate) size: String,
    pub(crate) source_video_id: Option<String>,
    pub(crate) character_id: Option<String>,
    pub(crate) metadata: Value,
    pub(crate) input_bytes: Vec<u8>,
}

pub(crate) async fn create_video(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    create_video_job(state, request, body, VideoOperation::Create, None).await
}

pub(crate) async fn edit_video(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    create_video_job(state, request, body, VideoOperation::Edit, None).await
}

pub(crate) async fn extend_video(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    create_video_job(state, request, body, VideoOperation::Extend, None).await
}

pub(crate) async fn remix_video(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Bytes,
) -> HttpResponse {
    create_video_job(
        state,
        request,
        body,
        VideoOperation::Remix,
        Some(path.into_inner()),
    )
    .await
}

async fn create_video_job(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
    operation: VideoOperation,
    path_source_video_id: Option<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let parsed = parse_video_request(operation, &request, &body)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(store_video_request(
            &state,
            parsed,
            operation,
            path_source_video_id,
        )?))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_videos(state: web::Data<AppState>, request: HttpRequest) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let videos = lock_videos(&state)?
            .values()
            .map(video_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "list",
            "data": videos
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_video(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let videos = lock_videos(&state)?;
        let record = videos
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("video '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(video_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_video(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let deleted = lock_videos(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "video.deleted",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_video_content(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let videos = lock_videos(&state)?;
        let record = videos
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("video '{id}' does not exist")))?;
        Ok::<_, ApiError>(
            HttpResponse::Ok()
                .content_type("video/mp4")
                .insert_header((
                    "content-disposition",
                    format!("attachment; filename=\"{id}.mp4\""),
                ))
                .body(record.content.clone()),
        )
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_video_character(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let parsed = parse_video_character_request(&request, &body)?;
        if let Some(source_video_id) = parsed.source_video_id.as_deref() {
            ensure_video_exists(&state, source_video_id)?;
        }
        let now = unix_seconds();
        let id = state.next_response_id("vchar");
        let record = VideoCharacterRecord {
            id: id.clone(),
            created_at: now,
            name: parsed.name,
            metadata: parsed.metadata,
            source_video_id: parsed.source_video_id,
        };
        lock_video_characters(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(video_character_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_video_character(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let characters = lock_video_characters(&state)?;
        let record = characters
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("video character '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(video_character_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn parse_video_json_request(
    operation: VideoOperation,
    body: &Value,
) -> Result<ParsedVideoRequest, ApiError> {
    if !body.is_object() {
        return Err(ApiError::bad_request("video request must be an object"));
    }
    let prompt = optional_string(body, "prompt")?
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default_prompt_for_operation(operation).to_string());
    if operation == VideoOperation::Create && prompt.trim().is_empty() {
        return Err(ApiError::bad_request("video create requires prompt"));
    }
    let model = optional_string(body, "model")?
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_VIDEO_MODEL.to_string());
    let seconds = normalize_video_seconds(
        optional_string(body, "seconds")?
            .as_deref()
            .unwrap_or(DEFAULT_VIDEO_SECONDS),
    )?;
    let size = normalize_video_size(
        optional_string(body, "size")?
            .as_deref()
            .unwrap_or(DEFAULT_VIDEO_SIZE),
    )?;
    let source_video_id = optional_string(body, "video_id")?
        .or_else(|| optional_string(body, "source_video_id").ok().flatten());
    let character_id = optional_string(body, "character_id")?;
    let metadata = request_metadata(body)?;
    let input_bytes = body
        .get("video")
        .or_else(|| body.get("input_video"))
        .or_else(|| body.get("content"))
        .or_else(|| body.get("data"))
        .and_then(Value::as_str)
        .map(|value| value.as_bytes().to_vec())
        .unwrap_or_default();
    Ok(ParsedVideoRequest {
        operation,
        model,
        prompt,
        seconds,
        size,
        source_video_id,
        character_id,
        metadata,
        input_bytes,
    })
}

pub(crate) fn create_video_batch_response(
    state: &AppState,
    operation: VideoOperation,
    body: &Value,
) -> Result<Value, ApiError> {
    let parsed = parse_video_json_request(operation, body)?;
    store_video_request(state, parsed, operation, None)
}

pub(crate) fn video_json(record: &VideoRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "completed_at": record.completed_at,
        "expires_at": record.expires_at,
        "model": record.model,
        "prompt": record.prompt,
        "seconds": record.seconds,
        "size": record.size,
        "status": record.status,
        "progress": record.progress,
        "error": record.error.clone().unwrap_or(Value::Null),
        "remixed_from_video_id": record.remixed_from_video_id,
        "character_id": record.character_id,
        "metadata": record.metadata,
        "nerva": {
            "operation": record.operation,
            "backend": "deterministic-placeholder"
        }
    })
}

pub(crate) fn video_character_json(record: &VideoCharacterRecord) -> Value {
    json!({
        "id": record.id,
        "created_at": record.created_at,
        "name": record.name,
        "metadata": record.metadata,
        "source_video_id": record.source_video_id
    })
}

fn store_video_request(
    state: &AppState,
    mut parsed: ParsedVideoRequest,
    operation: VideoOperation,
    path_source_video_id: Option<String>,
) -> Result<Value, ApiError> {
    if let Some(source_video_id) = path_source_video_id {
        ensure_video_exists(state, &source_video_id)?;
        parsed.source_video_id = Some(source_video_id);
    } else if let Some(source_video_id) = parsed.source_video_id.as_deref()
        && operation != VideoOperation::Create
    {
        ensure_video_exists(state, source_video_id)?;
    }
    if let Some(character_id) = parsed.character_id.as_deref() {
        ensure_character_exists(state, character_id)?;
    }
    let now = unix_seconds();
    let id = state.next_response_id("video");
    let content = placeholder_video_bytes(&parsed, &id, now);
    let record = VideoRecord {
        id: id.clone(),
        object: "video",
        created_at: now,
        completed_at: Some(now),
        expires_at: Some(now.saturating_add(VIDEO_ASSET_TTL_SECONDS)),
        model: parsed.model,
        prompt: parsed.prompt,
        seconds: parsed.seconds,
        size: parsed.size,
        status: "completed".to_string(),
        progress: 100,
        operation: parsed.operation.as_str().to_string(),
        remixed_from_video_id: parsed.source_video_id,
        character_id: parsed.character_id,
        metadata: parsed.metadata,
        error: None,
        content,
    };
    lock_videos(state)?.insert(id, record.clone());
    Ok(video_json(&record))
}

pub(crate) fn placeholder_video_bytes(
    parsed: &ParsedVideoRequest,
    id: &str,
    created_at: u64,
) -> Vec<u8> {
    let metadata = json!({
        "id": id,
        "created_at": created_at,
        "operation": parsed.operation.as_str(),
        "model": parsed.model,
        "prompt": parsed.prompt,
        "seconds": parsed.seconds,
        "size": parsed.size,
        "source_video_id": parsed.source_video_id,
        "character_id": parsed.character_id,
        "input_bytes": parsed.input_bytes.len()
    })
    .to_string();
    let mut out = Vec::new();
    out.extend_from_slice(&mp4_box(b"ftyp", b"isom\0\0\x02\0isomiso2mp41"));
    out.extend_from_slice(&mp4_box(b"free", b"NERVA placeholder video"));
    out.extend_from_slice(&mp4_box(b"mdat", metadata.as_bytes()));
    out
}

fn parse_video_request(
    operation: VideoOperation,
    request: &HttpRequest,
    body: &[u8],
) -> Result<ParsedVideoRequest, ApiError> {
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if content_type.starts_with("multipart/form-data") {
        let boundary = multipart_boundary(content_type)
            .ok_or_else(|| ApiError::bad_request("multipart video request is missing boundary"))?;
        return parse_multipart_video_request(operation, body, &boundary);
    }
    if content_type.starts_with("application/json") || looks_like_json(body) {
        let value: Value = serde_json::from_slice(body)
            .map_err(|err| ApiError::bad_request(format!("invalid video JSON request: {err}")))?;
        return parse_video_json_request(operation, &value);
    }
    let prompt = query_param(request.query_string(), "prompt")
        .unwrap_or_else(|| default_prompt_for_operation(operation).to_string());
    let value = json!({
        "prompt": prompt,
        "model": query_param(request.query_string(), "model").unwrap_or_else(|| DEFAULT_VIDEO_MODEL.to_string()),
        "seconds": query_param(request.query_string(), "seconds").unwrap_or_else(|| DEFAULT_VIDEO_SECONDS.to_string()),
        "size": query_param(request.query_string(), "size").unwrap_or_else(|| DEFAULT_VIDEO_SIZE.to_string()),
        "video_id": query_param(request.query_string(), "video_id"),
        "character_id": query_param(request.query_string(), "character_id"),
        "content": String::from_utf8_lossy(body)
    });
    parse_video_json_request(operation, &value)
}

fn parse_multipart_video_request(
    operation: VideoOperation,
    body: &[u8],
    boundary: &str,
) -> Result<ParsedVideoRequest, ApiError> {
    let text = String::from_utf8_lossy(body);
    let marker = format!("--{boundary}");
    let mut prompt = None;
    let mut model = None;
    let mut seconds = None;
    let mut size = None;
    let mut video_id = None;
    let mut character_id = None;
    let mut metadata = Value::Object(Default::default());
    let mut input_bytes = Vec::new();
    for raw_part in text.split(&marker) {
        let part = raw_part.trim_start_matches("\r\n");
        if part.is_empty() || part.starts_with("--") {
            continue;
        }
        let Some((headers, value)) = part.split_once("\r\n\r\n") else {
            continue;
        };
        let disposition = headers
            .lines()
            .find(|line| {
                line.to_ascii_lowercase()
                    .starts_with("content-disposition:")
            })
            .unwrap_or("");
        let Some(name) = disposition_param(disposition, "name") else {
            continue;
        };
        let value = value.strip_suffix("--").unwrap_or(value);
        let value = value.strip_suffix("\r\n").unwrap_or(value);
        match name.as_str() {
            "video" | "input_video" | "file" => input_bytes = value.as_bytes().to_vec(),
            "prompt" => prompt = Some(value.trim().to_string()),
            "model" => model = Some(value.trim().to_string()),
            "seconds" => seconds = Some(value.trim().to_string()),
            "size" => size = Some(value.trim().to_string()),
            "video_id" | "source_video_id" => video_id = Some(value.trim().to_string()),
            "character_id" => character_id = Some(value.trim().to_string()),
            "metadata" => {
                metadata = serde_json::from_str(value.trim())
                    .unwrap_or_else(|_| json!({"value": value.trim()}))
            }
            _ => {}
        }
    }
    parse_video_json_request(
        operation,
        &json!({
            "prompt": prompt,
            "model": model,
            "seconds": seconds,
            "size": size,
            "video_id": video_id,
            "character_id": character_id,
            "metadata": metadata,
            "content": String::from_utf8_lossy(&input_bytes)
        }),
    )
}

#[derive(Clone, Debug)]
struct ParsedVideoCharacterRequest {
    name: String,
    metadata: Value,
    source_video_id: Option<String>,
}

fn parse_video_character_request(
    request: &HttpRequest,
    body: &[u8],
) -> Result<ParsedVideoCharacterRequest, ApiError> {
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    let value = if content_type.starts_with("application/json") || looks_like_json(body) {
        serde_json::from_slice(body).map_err(|err| {
            ApiError::bad_request(format!("invalid video character JSON request: {err}"))
        })?
    } else {
        json!({
            "name": query_param(request.query_string(), "name").unwrap_or_else(|| "character".to_string()),
            "video_id": query_param(request.query_string(), "video_id")
        })
    };
    let name = optional_string(&value, "name")?
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "character".to_string());
    Ok(ParsedVideoCharacterRequest {
        name,
        metadata: request_metadata(&value)?,
        source_video_id: optional_string(&value, "video_id")?
            .or_else(|| optional_string(&value, "source_video_id").ok().flatten()),
    })
}

fn ensure_video_exists(state: &AppState, video_id: &str) -> Result<(), ApiError> {
    if lock_videos(state)?.contains_key(video_id) {
        Ok(())
    } else {
        Err(ApiError::not_found(format!(
            "video '{video_id}' does not exist"
        )))
    }
}

fn ensure_character_exists(state: &AppState, character_id: &str) -> Result<(), ApiError> {
    if lock_video_characters(state)?.contains_key(character_id) {
        Ok(())
    } else {
        Err(ApiError::not_found(format!(
            "video character '{character_id}' does not exist"
        )))
    }
}

fn lock_videos(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, VideoRecord>>, ApiError> {
    state
        .videos
        .lock()
        .map_err(|_| ApiError::internal("video registry lock poisoned"))
}

fn lock_video_characters(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, VideoCharacterRecord>>, ApiError> {
    state
        .video_characters
        .lock()
        .map_err(|_| ApiError::internal("video character registry lock poisoned"))
}

fn normalize_video_seconds(value: &str) -> Result<String, ApiError> {
    let value = value.trim();
    if matches!(value, "4" | "8" | "12") {
        Ok(value.to_string())
    } else {
        Err(ApiError::bad_request("video seconds must be 4, 8, or 12"))
    }
}

fn normalize_video_size(value: &str) -> Result<String, ApiError> {
    let value = value.trim();
    if matches!(value, "720x1280" | "1280x720" | "1024x1792" | "1792x1024") {
        Ok(value.to_string())
    } else {
        Err(ApiError::bad_request(
            "video size must be 720x1280, 1280x720, 1024x1792, or 1792x1024",
        ))
    }
}

fn optional_string(body: &Value, field: &'static str) -> Result<Option<String>, ApiError> {
    match body.get(field) {
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be a string"))),
    }
}

fn default_prompt_for_operation(operation: VideoOperation) -> &'static str {
    match operation {
        VideoOperation::Create => "",
        VideoOperation::Edit => "Edit the source video.",
        VideoOperation::Extend => "Extend the source video.",
        VideoOperation::Remix => "Remix the source video.",
    }
}

impl VideoOperation {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            VideoOperation::Create => "create",
            VideoOperation::Edit => "edit",
            VideoOperation::Extend => "extend",
            VideoOperation::Remix => "remix",
        }
    }
}

fn mp4_box(name: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len().saturating_add(8));
    let len = u32::try_from(data.len().saturating_add(8)).unwrap_or(u32::MAX);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(name);
    out.extend_from_slice(data);
    out
}

fn looks_like_json(body: &[u8]) -> bool {
    body.iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace())
        .is_some_and(|byte| byte == b'{' || byte == b'[')
}

fn multipart_boundary(content_type: &str) -> Option<String> {
    content_type
        .split(';')
        .map(str::trim)
        .find_map(|part| part.strip_prefix("boundary="))
        .map(|boundary| boundary.trim_matches('"').to_string())
        .filter(|boundary| !boundary.is_empty())
}

fn disposition_param(disposition: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    disposition
        .split(';')
        .map(str::trim)
        .find_map(|part| part.strip_prefix(&prefix))
        .map(|value| value.trim_matches('"').to_string())
        .filter(|value| !value.is_empty())
}

fn query_param(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|part| {
        let (key, value) = part.split_once('=')?;
        (percent_decode_query(key) == name).then(|| percent_decode_query(value))
    })
}

fn percent_decode_query(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                out.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let hi = hex_value(bytes[index + 1]);
                let lo = hex_value(bytes[index + 2]);
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    out.push((hi << 4) | lo);
                    index += 3;
                } else {
                    out.push(bytes[index]);
                    index += 1;
                }
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
