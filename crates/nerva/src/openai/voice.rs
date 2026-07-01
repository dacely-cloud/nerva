use std::collections::HashMap;
use std::sync::atomic::Ordering;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, AudioVoiceRecord, VoiceConsentRecord, authorize, disposition_param,
    looks_like_json, multipart_boundary, query_param, unix_seconds,
};

const DEFAULT_AUDIO_CONTENT_TYPE: &str = "application/octet-stream";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParsedAudioVoiceRequest {
    pub(crate) name: String,
    pub(crate) consent: String,
    pub(crate) filename: String,
    pub(crate) content_type: String,
    pub(crate) content: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParsedVoiceConsentRequest {
    pub(crate) name: String,
    pub(crate) language: String,
    pub(crate) filename: String,
    pub(crate) content_type: String,
    pub(crate) content: Vec<u8>,
}

pub(crate) async fn create_audio_voice(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        let parsed = parse_audio_voice_request(&request, &body)?;
        let id = state.next_response_id("voice");
        let record = AudioVoiceRecord {
            id: id.clone(),
            object: "audio.voice",
            created_at: unix_seconds(),
            name: parsed.name,
            consent: parsed.consent,
            sample_filename: parsed.filename,
            sample_content_type: parsed.content_type,
            sample_bytes: parsed.content,
        };
        lock_audio_voices(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(audio_voice_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_voice_consent(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        let parsed = parse_voice_consent_request(&request, &body)?;
        let id = state.next_response_id("vconsent");
        let record = VoiceConsentRecord {
            id: id.clone(),
            object: "audio.voice_consent",
            created_at: unix_seconds(),
            language: parsed.language,
            name: parsed.name,
            recording_filename: parsed.filename,
            recording_content_type: parsed.content_type,
            recording_bytes: parsed.content,
        };
        lock_voice_consents(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(voice_consent_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_voice_consents(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let query = parse_list_query(request.query_string())?;
        let consents = lock_voice_consents(&state)?;
        let mut records = consents.values().collect::<Vec<_>>();
        records.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.id.cmp(&right.id))
        });
        if let Some(after) = query.after.as_deref() {
            if let Some(position) = records.iter().position(|record| record.id == after) {
                records = records.into_iter().skip(position + 1).collect();
            }
        }
        let has_more = records.len() > query.limit;
        records.truncate(query.limit);
        Ok::<_, ApiError>(HttpResponse::Ok().json(voice_consent_list_json(records, has_more)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_voice_consent(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let consents = lock_voice_consents(&state)?;
        let record = consents
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("voice consent '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(voice_consent_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn update_voice_consent(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Bytes,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let name = parse_voice_consent_update(&request, &body)?;
        let mut consents = lock_voice_consents(&state)?;
        let record = consents
            .get_mut(&id)
            .ok_or_else(|| ApiError::not_found(format!("voice consent '{id}' does not exist")))?;
        record.name = name;
        Ok::<_, ApiError>(HttpResponse::Ok().json(voice_consent_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_voice_consent(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let deleted = lock_voice_consents(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "audio.voice_consent",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn parse_audio_voice_request(
    request: &HttpRequest,
    body: &[u8],
) -> Result<ParsedAudioVoiceRequest, ApiError> {
    let content_type = request_content_type(request);
    if content_type.starts_with("multipart/form-data") {
        let boundary = multipart_boundary(content_type)
            .ok_or_else(|| ApiError::bad_request("multipart voice request is missing boundary"))?;
        return normalize_audio_voice_fields(parse_multipart_voice_fields(
            body,
            &boundary,
            &["audio_sample", "sample", "file"],
        )?);
    }
    if content_type.starts_with("application/json") || looks_like_json(body) {
        let value: Value = serde_json::from_slice(body)
            .map_err(|err| ApiError::bad_request(format!("invalid voice JSON request: {err}")))?;
        return parse_audio_voice_json_request(&value);
    }
    normalize_audio_voice_fields(ParsedVoiceFields {
        text: HashMap::from([
            (
                "name".to_string(),
                query_param(request.query_string(), "name"),
            ),
            (
                "consent".to_string(),
                query_param(request.query_string(), "consent"),
            ),
        ]),
        upload: Some(ParsedVoiceUpload {
            filename: query_param(request.query_string(), "filename")
                .unwrap_or_else(|| "voice.bin".to_string()),
            content_type: content_type
                .trim()
                .is_empty()
                .then_some(DEFAULT_AUDIO_CONTENT_TYPE)
                .unwrap_or(content_type)
                .to_string(),
            content: body.to_vec(),
        }),
    })
}

pub(crate) fn parse_audio_voice_json_request(
    body: &Value,
) -> Result<ParsedAudioVoiceRequest, ApiError> {
    let content = request_json_audio_content(
        body,
        &["audio_sample", "sample", "file", "content", "data"],
        "voice JSON request requires audio_sample, sample, file, content, or data",
    )?;
    normalize_audio_voice_fields(ParsedVoiceFields {
        text: HashMap::from([
            (
                "name".to_string(),
                body.get("name").and_then(Value::as_str).map(str::to_string),
            ),
            (
                "consent".to_string(),
                body.get("consent")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            ),
        ]),
        upload: Some(ParsedVoiceUpload {
            filename: json_string(body, "filename").unwrap_or_else(|| "voice.json".to_string()),
            content_type: json_string(body, "content_type")
                .or_else(|| json_string(body, "mime_type"))
                .unwrap_or_else(|| DEFAULT_AUDIO_CONTENT_TYPE.to_string()),
            content,
        }),
    })
}

pub(crate) fn parse_voice_consent_request(
    request: &HttpRequest,
    body: &[u8],
) -> Result<ParsedVoiceConsentRequest, ApiError> {
    let content_type = request_content_type(request);
    if content_type.starts_with("multipart/form-data") {
        let boundary = multipart_boundary(content_type).ok_or_else(|| {
            ApiError::bad_request("multipart voice consent request is missing boundary")
        })?;
        return normalize_voice_consent_fields(parse_multipart_voice_fields(
            body,
            &boundary,
            &["recording", "audio", "file"],
        )?);
    }
    if content_type.starts_with("application/json") || looks_like_json(body) {
        let value: Value = serde_json::from_slice(body).map_err(|err| {
            ApiError::bad_request(format!("invalid voice consent JSON request: {err}"))
        })?;
        return parse_voice_consent_json_request(&value);
    }
    normalize_voice_consent_fields(ParsedVoiceFields {
        text: HashMap::from([
            (
                "name".to_string(),
                query_param(request.query_string(), "name"),
            ),
            (
                "language".to_string(),
                query_param(request.query_string(), "language"),
            ),
        ]),
        upload: Some(ParsedVoiceUpload {
            filename: query_param(request.query_string(), "filename")
                .unwrap_or_else(|| "voice-consent.bin".to_string()),
            content_type: content_type
                .trim()
                .is_empty()
                .then_some(DEFAULT_AUDIO_CONTENT_TYPE)
                .unwrap_or(content_type)
                .to_string(),
            content: body.to_vec(),
        }),
    })
}

pub(crate) fn parse_voice_consent_json_request(
    body: &Value,
) -> Result<ParsedVoiceConsentRequest, ApiError> {
    let content = request_json_audio_content(
        body,
        &["recording", "audio", "file", "content", "data"],
        "voice consent JSON request requires recording, audio, file, content, or data",
    )?;
    normalize_voice_consent_fields(ParsedVoiceFields {
        text: HashMap::from([
            (
                "name".to_string(),
                body.get("name").and_then(Value::as_str).map(str::to_string),
            ),
            (
                "language".to_string(),
                body.get("language")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            ),
        ]),
        upload: Some(ParsedVoiceUpload {
            filename: json_string(body, "filename")
                .unwrap_or_else(|| "voice-consent.json".to_string()),
            content_type: json_string(body, "content_type")
                .or_else(|| json_string(body, "mime_type"))
                .unwrap_or_else(|| DEFAULT_AUDIO_CONTENT_TYPE.to_string()),
            content,
        }),
    })
}

pub(crate) fn audio_voice_json(record: &AudioVoiceRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "name": record.name,
        "nerva": {
            "consent": record.consent,
            "sample": {
                "filename": record.sample_filename,
                "content_type": record.sample_content_type,
                "bytes": record.sample_bytes.len()
            }
        }
    })
}

pub(crate) fn voice_consent_json(record: &VoiceConsentRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "language": record.language,
        "name": record.name,
        "nerva": {
            "recording": {
                "filename": record.recording_filename,
                "content_type": record.recording_content_type,
                "bytes": record.recording_bytes.len()
            }
        }
    })
}

pub(crate) fn voice_consent_list_json(records: Vec<&VoiceConsentRecord>, has_more: bool) -> Value {
    let data = records
        .iter()
        .map(|record| voice_consent_json(record))
        .collect::<Vec<_>>();
    json!({
        "object": "list",
        "data": data,
        "first_id": records.first().map(|record| record.id.as_str()),
        "last_id": records.last().map(|record| record.id.as_str()),
        "has_more": has_more
    })
}

#[derive(Clone, Debug)]
struct ParsedVoiceFields {
    text: HashMap<String, Option<String>>,
    upload: Option<ParsedVoiceUpload>,
}

#[derive(Clone, Debug)]
struct ParsedVoiceUpload {
    filename: String,
    content_type: String,
    content: Vec<u8>,
}

#[derive(Clone, Debug)]
struct ListQuery {
    after: Option<String>,
    limit: usize,
}

fn normalize_audio_voice_fields(
    fields: ParsedVoiceFields,
) -> Result<ParsedAudioVoiceRequest, ApiError> {
    let upload = require_upload(
        fields.upload,
        "audio voice request requires non-empty audio_sample",
    )?;
    Ok(ParsedAudioVoiceRequest {
        name: required_field(&fields.text, "name")?,
        consent: required_field(&fields.text, "consent")?,
        filename: upload.filename,
        content_type: upload.content_type,
        content: upload.content,
    })
}

fn normalize_voice_consent_fields(
    fields: ParsedVoiceFields,
) -> Result<ParsedVoiceConsentRequest, ApiError> {
    let upload = require_upload(
        fields.upload,
        "voice consent request requires non-empty recording",
    )?;
    Ok(ParsedVoiceConsentRequest {
        name: required_field(&fields.text, "name")?,
        language: required_field(&fields.text, "language")?,
        filename: upload.filename,
        content_type: upload.content_type,
        content: upload.content,
    })
}

fn parse_multipart_voice_fields(
    body: &[u8],
    boundary: &str,
    upload_names: &[&str],
) -> Result<ParsedVoiceFields, ApiError> {
    let text = String::from_utf8_lossy(body);
    let marker = format!("--{boundary}");
    let mut fields = ParsedVoiceFields {
        text: HashMap::new(),
        upload: None,
    };
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
        if upload_names.iter().any(|candidate| *candidate == name) {
            fields.upload = Some(ParsedVoiceUpload {
                filename: disposition_param(disposition, "filename")
                    .unwrap_or_else(|| format!("{name}.bin")),
                content_type: header_content_type(headers)
                    .unwrap_or_else(|| DEFAULT_AUDIO_CONTENT_TYPE.to_string()),
                content: value.as_bytes().to_vec(),
            });
        } else {
            fields.text.insert(name, Some(value.trim().to_string()));
        }
    }
    Ok(fields)
}

fn parse_voice_consent_update(request: &HttpRequest, body: &[u8]) -> Result<String, ApiError> {
    let content_type = request_content_type(request);
    let name = if content_type.starts_with("application/json") || looks_like_json(body) {
        let value: Value = serde_json::from_slice(body).map_err(|err| {
            ApiError::bad_request(format!("invalid voice consent update JSON: {err}"))
        })?;
        json_string(&value, "name")
    } else {
        query_param(request.query_string(), "name")
    };
    require_nonempty(name, "name")
}

fn parse_list_query(query: &str) -> Result<ListQuery, ApiError> {
    let limit = query_param(query, "limit")
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| ApiError::bad_request("limit must be an integer"))
        })
        .transpose()?
        .unwrap_or(20);
    if !(1..=100).contains(&limit) {
        return Err(ApiError::bad_request("limit must be between 1 and 100"));
    }
    Ok(ListQuery {
        after: query_param(query, "after").filter(|value| !value.trim().is_empty()),
        limit,
    })
}

fn request_json_audio_content(
    body: &Value,
    fields: &[&str],
    missing_message: &'static str,
) -> Result<Vec<u8>, ApiError> {
    fields
        .iter()
        .find_map(|field| body.get(*field).and_then(Value::as_str))
        .ok_or_else(|| ApiError::bad_request(missing_message))
        .map(|value| value.as_bytes().to_vec())
}

fn require_upload(
    upload: Option<ParsedVoiceUpload>,
    missing_message: &'static str,
) -> Result<ParsedVoiceUpload, ApiError> {
    let upload = upload.ok_or_else(|| ApiError::bad_request(missing_message))?;
    if upload.content.is_empty() {
        return Err(ApiError::bad_request(missing_message));
    }
    Ok(upload)
}

fn required_field(
    fields: &HashMap<String, Option<String>>,
    field: &'static str,
) -> Result<String, ApiError> {
    require_nonempty(fields.get(field).cloned().flatten(), field)
}

fn require_nonempty(value: Option<String>, field: &'static str) -> Result<String, ApiError> {
    match value {
        Some(value) if !value.trim().is_empty() => Ok(value),
        Some(_) => Err(ApiError::bad_request(format!("{field} must not be empty"))),
        None => Err(ApiError::bad_request(format!("{field} is required"))),
    }
}

fn json_string(body: &Value, field: &'static str) -> Option<String> {
    body.get(field).and_then(Value::as_str).map(str::to_string)
}

fn request_content_type(request: &HttpRequest) -> &str {
    request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
}

fn header_content_type(headers: &str) -> Option<String> {
    headers
        .lines()
        .find_map(|line| line.strip_prefix("Content-Type:"))
        .or_else(|| {
            headers
                .lines()
                .find_map(|line| line.strip_prefix("content-type:"))
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn lock_audio_voices(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, AudioVoiceRecord>>, ApiError> {
    state
        .audio_voices
        .lock()
        .map_err(|_| ApiError::internal("audio voice registry lock poisoned"))
}

fn lock_voice_consents(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, VoiceConsentRecord>>, ApiError> {
    state
        .voice_consents
        .lock()
        .map_err(|_| ApiError::internal("voice consent registry lock poisoned"))
}
