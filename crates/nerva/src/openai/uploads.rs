use std::collections::{HashMap, HashSet};

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, FileRecord, UploadPartRecord, UploadRecord, authorize, file_json,
    lock_files, unix_seconds,
};

const MAX_UPLOAD_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_UPLOAD_PART_BYTES: usize = 64 * 1024 * 1024;
const UPLOAD_EXPIRES_AFTER_SECONDS: u64 = 3600;

pub(crate) async fn create_upload(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let fields = parse_upload_create_request(&body)?;
        let now = unix_seconds();
        let id = state.next_response_id("upload");
        let record = UploadRecord {
            id: id.clone(),
            object: "upload",
            bytes: fields.bytes,
            created_at: now,
            expires_at: now + UPLOAD_EXPIRES_AFTER_SECONDS,
            filename: fields.filename,
            purpose: fields.purpose,
            mime_type: fields.mime_type,
            status: "pending".to_string(),
            parts: Vec::new(),
            file_id: None,
        };
        lock_uploads(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(upload_json(&record, None)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_upload_part(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Bytes,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let upload_id = path.into_inner();
        let content = parse_upload_part_body(&request, &body)?;
        if content.is_empty() {
            return Err(ApiError::bad_request(
                "upload part content must not be empty",
            ));
        }
        if content.len() > MAX_UPLOAD_PART_BYTES {
            return Err(ApiError::bad_request(format!(
                "upload part exceeds {} bytes",
                MAX_UPLOAD_PART_BYTES
            )));
        }
        let mut uploads = lock_uploads(&state)?;
        let record = uploads
            .get_mut(&upload_id)
            .ok_or_else(|| ApiError::not_found(format!("upload '{upload_id}' does not exist")))?;
        ensure_upload_pending(record)?;
        let uploaded_bytes = record
            .parts
            .iter()
            .map(|part| part.bytes as u64)
            .sum::<u64>();
        let next_total = uploaded_bytes.saturating_add(content.len() as u64);
        if next_total > record.bytes {
            return Err(ApiError::bad_request(format!(
                "upload parts exceed declared upload size of {} bytes",
                record.bytes
            )));
        }
        let part = UploadPartRecord {
            id: state.next_response_id("part"),
            object: "upload.part",
            created_at: unix_seconds(),
            upload_id: upload_id.clone(),
            bytes: content.len(),
            content,
        };
        record.parts.push(part.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(upload_part_json(&part)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn complete_upload(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let upload_id = path.into_inner();
        let part_ids = request_part_ids(&body)?;
        let mut uploads = lock_uploads(&state)?;
        let record = uploads
            .get_mut(&upload_id)
            .ok_or_else(|| ApiError::not_found(format!("upload '{upload_id}' does not exist")))?;
        ensure_upload_pending(record)?;
        let content = upload_content_for_part_ids(record, &part_ids)?;
        if content.len() as u64 != record.bytes {
            return Err(ApiError::bad_request(format!(
                "completed upload has {} bytes, expected {} bytes",
                content.len(),
                record.bytes
            )));
        }
        let file = FileRecord {
            id: state.next_response_id("file"),
            object: "file",
            bytes: content.len(),
            created_at: unix_seconds(),
            filename: record.filename.clone(),
            purpose: record.purpose.clone(),
            status: "processed",
            content,
        };
        lock_files(&state)?.insert(file.id.clone(), file.clone());
        record.status = "completed".to_string();
        record.file_id = Some(file.id.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(upload_json(record, Some(&file))))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn cancel_upload(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let upload_id = path.into_inner();
        let mut uploads = lock_uploads(&state)?;
        let record = uploads
            .get_mut(&upload_id)
            .ok_or_else(|| ApiError::not_found(format!("upload '{upload_id}' does not exist")))?;
        if record.status == "completed" {
            return Err(ApiError::bad_request(
                "completed uploads cannot be cancelled",
            ));
        }
        record.status = "cancelled".to_string();
        Ok::<_, ApiError>(HttpResponse::Ok().json(upload_json(record, None)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

#[derive(Clone, Debug)]
pub(crate) struct UploadCreateFields {
    pub(crate) filename: String,
    pub(crate) purpose: String,
    pub(crate) bytes: u64,
    pub(crate) mime_type: String,
}

pub(crate) fn parse_upload_create_request(body: &Value) -> Result<UploadCreateFields, ApiError> {
    let filename = required_nonempty_string(body, "filename")?;
    let purpose = required_nonempty_string(body, "purpose")?;
    let bytes = body
        .get("bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| ApiError::bad_request("upload create requires positive integer bytes"))?;
    if bytes == 0 {
        return Err(ApiError::bad_request("upload bytes must be non-zero"));
    }
    if bytes > MAX_UPLOAD_BYTES {
        return Err(ApiError::bad_request(format!(
            "upload exceeds maximum size of {MAX_UPLOAD_BYTES} bytes"
        )));
    }
    let mime_type = body
        .get("mime_type")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("application/octet-stream")
        .to_string();
    Ok(UploadCreateFields {
        filename,
        purpose,
        bytes,
        mime_type,
    })
}

pub(crate) fn parse_upload_part_body(
    request: &HttpRequest,
    body: &[u8],
) -> Result<Vec<u8>, ApiError> {
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if content_type.starts_with("multipart/form-data") {
        let boundary = multipart_boundary(content_type)
            .ok_or_else(|| ApiError::bad_request("multipart upload part is missing boundary"))?;
        return parse_multipart_upload_part(body, &boundary);
    }
    if content_type.starts_with("application/json") || looks_like_json(body) {
        let value: Value = serde_json::from_slice(body)
            .map_err(|err| ApiError::bad_request(format!("invalid upload part JSON: {err}")))?;
        return value
            .get("data")
            .or_else(|| value.get("content"))
            .and_then(Value::as_str)
            .map(|text| text.as_bytes().to_vec())
            .ok_or_else(|| ApiError::bad_request("JSON upload part requires data or content"));
    }
    Ok(body.to_vec())
}

pub(crate) fn upload_content_for_part_ids(
    record: &UploadRecord,
    part_ids: &[String],
) -> Result<Vec<u8>, ApiError> {
    if part_ids.is_empty() {
        return Err(ApiError::bad_request("complete upload requires part_ids"));
    }
    let mut seen = HashSet::new();
    let mut content = Vec::new();
    for part_id in part_ids {
        if !seen.insert(part_id) {
            return Err(ApiError::bad_request(format!(
                "duplicate upload part id '{part_id}'"
            )));
        }
        let part = record
            .parts
            .iter()
            .find(|part| part.id == *part_id)
            .ok_or_else(|| {
                ApiError::bad_request(format!(
                    "upload part '{part_id}' does not belong to upload '{}'",
                    record.id
                ))
            })?;
        content.extend_from_slice(&part.content);
    }
    Ok(content)
}

fn request_part_ids(body: &Value) -> Result<Vec<String>, ApiError> {
    let Some(values) = body.get("part_ids").and_then(Value::as_array) else {
        return Err(ApiError::bad_request("complete upload requires part_ids"));
    };
    let mut ids = Vec::with_capacity(values.len());
    for value in values {
        let Some(id) = value.as_str().filter(|id| !id.trim().is_empty()) else {
            return Err(ApiError::bad_request(
                "complete upload part_ids must be non-empty strings",
            ));
        };
        ids.push(id.to_string());
    }
    Ok(ids)
}

fn ensure_upload_pending(record: &mut UploadRecord) -> Result<(), ApiError> {
    if record.status == "pending" && unix_seconds() >= record.expires_at {
        record.status = "expired".to_string();
    }
    if record.status != "pending" {
        return Err(ApiError::bad_request(format!(
            "upload '{}' is {}",
            record.id, record.status
        )));
    }
    Ok(())
}

fn parse_multipart_upload_part(body: &[u8], boundary: &str) -> Result<Vec<u8>, ApiError> {
    let marker = format!("--{boundary}").into_bytes();
    let mut search_start = 0usize;
    while let Some(marker_offset) = find_subslice(&body[search_start..], &marker) {
        let mut part_start = search_start + marker_offset + marker.len();
        if body
            .get(part_start..part_start.saturating_add(2))
            .is_some_and(|bytes| bytes == b"--")
        {
            break;
        }
        if body
            .get(part_start..part_start.saturating_add(2))
            .is_some_and(|bytes| bytes == b"\r\n")
        {
            part_start += 2;
        }
        let Some(next_marker_offset) = find_subslice(&body[part_start..], &marker) else {
            break;
        };
        let mut part = &body[part_start..part_start + next_marker_offset];
        if part.ends_with(b"\r\n") {
            part = &part[..part.len() - 2];
        }
        let Some(header_end) = find_subslice(part, b"\r\n\r\n") else {
            search_start = part_start + next_marker_offset;
            continue;
        };
        let headers = String::from_utf8_lossy(&part[..header_end]);
        let value = &part[header_end + 4..];
        let disposition = headers
            .lines()
            .find(|line| {
                line.to_ascii_lowercase()
                    .starts_with("content-disposition:")
            })
            .unwrap_or("");
        let Some(name) = disposition_param(disposition, "name") else {
            search_start = part_start + next_marker_offset;
            continue;
        };
        if name == "data" || name == "file" {
            return Ok(value.to_vec());
        }
        search_start = part_start + next_marker_offset;
    }
    Err(ApiError::bad_request(
        "multipart upload part requires data or file",
    ))
}

fn upload_json(record: &UploadRecord, file: Option<&FileRecord>) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "bytes": record.bytes,
        "created_at": record.created_at,
        "expires_at": record.expires_at,
        "filename": record.filename,
        "purpose": record.purpose,
        "mime_type": record.mime_type,
        "status": record.status,
        "file": file.map(file_json),
        "file_id": record.file_id,
        "parts": {
            "object": "list",
            "data": record.parts.iter().map(upload_part_json).collect::<Vec<_>>()
        }
    })
}

fn upload_part_json(record: &UploadPartRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "upload_id": record.upload_id,
        "bytes": record.bytes
    })
}

fn lock_uploads(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, UploadRecord>>, ApiError> {
    state
        .uploads
        .lock()
        .map_err(|_| ApiError::internal("upload registry lock poisoned"))
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

fn required_nonempty_string(body: &Value, name: &str) -> Result<String, ApiError> {
    body.get(name)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| ApiError::bad_request(format!("upload create requires {name}")))
}

fn looks_like_json(body: &[u8]) -> bool {
    body.iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace())
        .is_some_and(|byte| byte == b'{' || byte == b'[')
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}
