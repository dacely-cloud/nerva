use std::collections::HashMap;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{ApiError, AppState, FileRecord, authorize, unix_seconds};

pub(crate) async fn create_file(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let upload = parse_file_upload(&request, &body)?;
        if upload.content.is_empty() {
            return Err(ApiError::bad_request(
                "uploaded file content must not be empty",
            ));
        }
        let id = state.next_response_id("file");
        let record = FileRecord {
            id: id.clone(),
            object: "file",
            bytes: upload.content.len(),
            created_at: unix_seconds(),
            filename: upload.filename,
            purpose: upload.purpose,
            status: "processed",
            content: upload.content,
        };
        lock_files(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(file_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_files(state: web::Data<AppState>, request: HttpRequest) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let purpose = query_param(request.query_string(), "purpose");
        let files = lock_files(&state)?
            .values()
            .filter(|file| {
                purpose
                    .as_deref()
                    .is_none_or(|purpose| purpose == file.purpose)
            })
            .map(file_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "list",
            "data": files
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_file(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let files = lock_files(&state)?;
        let record = files
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("file '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(file_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_file(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let deleted = lock_files(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "file",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_file_content(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let files = lock_files(&state)?;
        let record = files
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("file '{id}' does not exist")))?;
        Ok::<_, ApiError>(
            HttpResponse::Ok()
                .content_type(file_content_type(&record.filename))
                .body(record.content.clone()),
        )
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

#[derive(Clone, Debug)]
pub(crate) struct ParsedFileUpload {
    pub(crate) filename: String,
    pub(crate) purpose: String,
    pub(crate) content: Vec<u8>,
}

fn parse_file_upload(request: &HttpRequest, body: &[u8]) -> Result<ParsedFileUpload, ApiError> {
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if content_type.starts_with("multipart/form-data") {
        let boundary = multipart_boundary(content_type)
            .ok_or_else(|| ApiError::bad_request("multipart upload is missing boundary"))?;
        return parse_multipart_file_upload(body, &boundary);
    }
    if content_type.starts_with("application/json") || looks_like_json(body) {
        let value: Value = serde_json::from_slice(body)
            .map_err(|err| ApiError::bad_request(format!("invalid file JSON upload: {err}")))?;
        let content = value
            .get("content")
            .or_else(|| value.get("data"))
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::bad_request("JSON file upload requires content or data"))?
            .as_bytes()
            .to_vec();
        let filename = value
            .get("filename")
            .and_then(Value::as_str)
            .unwrap_or("upload.jsonl")
            .to_string();
        let purpose = value
            .get("purpose")
            .and_then(Value::as_str)
            .unwrap_or("batch")
            .to_string();
        return Ok(ParsedFileUpload {
            filename,
            purpose,
            content,
        });
    }
    let filename =
        query_param(request.query_string(), "filename").unwrap_or_else(|| "upload.bin".to_string());
    let purpose = query_param(request.query_string(), "purpose").unwrap_or_else(|| "batch".into());
    Ok(ParsedFileUpload {
        filename,
        purpose,
        content: body.to_vec(),
    })
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

pub(crate) fn parse_multipart_file_upload(
    body: &[u8],
    boundary: &str,
) -> Result<ParsedFileUpload, ApiError> {
    let text = String::from_utf8_lossy(body);
    let marker = format!("--{boundary}");
    let mut filename = None;
    let mut purpose = None;
    let mut content = None;
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
            "purpose" => purpose = Some(value.trim().to_string()),
            "file" => {
                filename = disposition_param(disposition, "filename")
                    .or_else(|| query_filename_from_headers(headers).map(str::to_string));
                content = Some(value.as_bytes().to_vec());
            }
            _ => {}
        }
    }
    Ok(ParsedFileUpload {
        filename: filename.unwrap_or_else(|| "upload.jsonl".to_string()),
        purpose: purpose.unwrap_or_else(|| "batch".to_string()),
        content: content.ok_or_else(|| ApiError::bad_request("multipart upload requires file"))?,
    })
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

fn query_filename_from_headers(headers: &str) -> Option<&str> {
    headers
        .lines()
        .find_map(|line| line.strip_prefix("filename:"))
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn file_content_type(filename: &str) -> &'static str {
    if filename.ends_with(".jsonl") {
        "application/jsonl"
    } else if filename.ends_with(".json") {
        "application/json"
    } else if filename.ends_with(".txt") {
        "text/plain; charset=utf-8"
    } else {
        "application/octet-stream"
    }
}

fn query_param(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|part| {
        let (key, value) = part.split_once('=')?;
        (percent_decode_query(key) == name).then(|| percent_decode_query(value))
    })
}

pub(crate) fn percent_decode_query(value: &str) -> String {
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

pub(crate) fn insert_generated_file(
    state: &AppState,
    filename: String,
    purpose: &str,
    content: Vec<u8>,
) -> Result<String, ApiError> {
    let id = state.next_response_id("file");
    let record = FileRecord {
        id: id.clone(),
        object: "file",
        bytes: content.len(),
        created_at: unix_seconds(),
        filename,
        purpose: purpose.to_string(),
        status: "processed",
        content,
    };
    lock_files(state)?.insert(id.clone(), record);
    Ok(id)
}

pub(crate) fn lock_files(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, FileRecord>>, ApiError> {
    state
        .files
        .lock()
        .map_err(|_| ApiError::internal("file registry lock poisoned"))
}

fn file_json(record: &FileRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "bytes": record.bytes,
        "created_at": record.created_at,
        "filename": record.filename,
        "purpose": record.purpose,
        "status": record.status,
        "status_details": null
    })
}
