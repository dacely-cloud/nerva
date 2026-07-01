use std::collections::HashMap;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, SkillRecord, SkillVersionRecord, authorize, parse_file_upload,
    request_metadata, request_optional_string, unix_seconds,
};

pub(crate) async fn create_skill(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let parsed = parse_skill_payload(&request, &body)?;
        let now = unix_seconds();
        let skill_id = state.next_response_id("skill");
        let version_id = state.next_response_id("skillver");
        let version = SkillVersionRecord {
            id: version_id.clone(),
            object: "skill.version",
            created_at: now,
            skill_id: skill_id.clone(),
            name: parsed.name.clone(),
            description: parsed.description.clone(),
            status: "ready".to_string(),
            content_type: parsed.content_type,
            content: parsed.content,
            metadata: parsed.metadata.clone(),
        };
        let mut versions = HashMap::new();
        versions.insert(version_id.clone(), version);
        let record = SkillRecord {
            id: skill_id.clone(),
            object: "skill",
            created_at: now,
            updated_at: now,
            name: parsed.name,
            description: parsed.description,
            metadata: parsed.metadata,
            status: "ready".to_string(),
            current_version_id: Some(version_id),
            versions,
        };
        lock_skills(&state)?.insert(skill_id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(skill_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_skills(state: web::Data<AppState>, request: HttpRequest) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let data = lock_skills(&state)?
            .values()
            .map(skill_json)
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

pub(crate) async fn get_skill(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let skills = lock_skills(&state)?;
        let record = skills
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("skill '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(skill_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn update_skill(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let name = request_optional_string(&body, "name")?;
        let description = request_optional_string(&body, "description")?;
        let metadata = request_metadata(&body)?;
        let mut skills = lock_skills(&state)?;
        let record = skills
            .get_mut(&id)
            .ok_or_else(|| ApiError::not_found(format!("skill '{id}' does not exist")))?;
        if let Some(name) = name {
            record.name = name;
        }
        if body.get("description").is_some() {
            record.description = description;
        }
        if body.get("metadata").is_some() {
            record.metadata = metadata;
        }
        record.updated_at = unix_seconds();
        Ok::<_, ApiError>(HttpResponse::Ok().json(skill_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_skill(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let deleted = lock_skills(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "skill.deleted",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_skill_content(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let version = current_skill_version(&state, &id)?;
        Ok::<_, ApiError>(
            HttpResponse::Ok()
                .content_type(version.content_type)
                .body(version.content),
        )
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_skill_version(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Bytes,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let skill_id = path.into_inner();
        let parsed = parse_skill_payload(&request, &body)?;
        let version = SkillVersionRecord {
            id: state.next_response_id("skillver"),
            object: "skill.version",
            created_at: unix_seconds(),
            skill_id: skill_id.clone(),
            name: parsed.name,
            description: parsed.description,
            status: "ready".to_string(),
            content_type: parsed.content_type,
            content: parsed.content,
            metadata: parsed.metadata,
        };
        let mut skills = lock_skills(&state)?;
        let record = skills
            .get_mut(&skill_id)
            .ok_or_else(|| ApiError::not_found(format!("skill '{skill_id}' does not exist")))?;
        record.current_version_id = Some(version.id.clone());
        record.updated_at = unix_seconds();
        record.versions.insert(version.id.clone(), version.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(skill_version_json(&version)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_skill_versions(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let skill_id = path.into_inner();
        let skills = lock_skills(&state)?;
        let record = skills
            .get(&skill_id)
            .ok_or_else(|| ApiError::not_found(format!("skill '{skill_id}' does not exist")))?;
        let data = record
            .versions
            .values()
            .map(skill_version_json)
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

pub(crate) async fn get_skill_version(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (skill_id, version_id) = path.into_inner();
        let version = skill_version(&state, &skill_id, &version_id)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(skill_version_json(&version)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_skill_version_content(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (skill_id, version_id) = path.into_inner();
        let version = skill_version(&state, &skill_id, &version_id)?;
        Ok::<_, ApiError>(
            HttpResponse::Ok()
                .content_type(version.content_type)
                .body(version.content),
        )
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_skill_version(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (skill_id, version_id) = path.into_inner();
        let mut skills = lock_skills(&state)?;
        let record = skills
            .get_mut(&skill_id)
            .ok_or_else(|| ApiError::not_found(format!("skill '{skill_id}' does not exist")))?;
        let deleted = record.versions.remove(&version_id).is_some();
        if record.current_version_id.as_deref() == Some(&version_id) {
            record.current_version_id = record.versions.keys().next().cloned();
        }
        record.updated_at = unix_seconds();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": version_id,
            "object": "skill.version.deleted",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

#[derive(Clone, Debug)]
pub(crate) struct ParsedSkillPayload {
    pub(crate) name: String,
    pub(crate) description: Option<String>,
    pub(crate) metadata: Value,
    pub(crate) content_type: String,
    pub(crate) content: Vec<u8>,
}

pub(crate) fn parse_skill_json_payload(body: &Value) -> Result<ParsedSkillPayload, ApiError> {
    let name = body
        .get("name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .ok_or_else(|| ApiError::bad_request("skill requires name"))?;
    let description = optional_string_field(body, "description")?;
    let metadata = request_metadata(body)?;
    let content = body
        .get("content")
        .or_else(|| body.get("data"))
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("skill requires content or data"))?
        .as_bytes()
        .to_vec();
    let content_type = body
        .get("content_type")
        .or_else(|| body.get("mime_type"))
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("text/markdown; charset=utf-8")
        .to_string();
    Ok(ParsedSkillPayload {
        name,
        description,
        metadata,
        content_type,
        content,
    })
}

fn parse_skill_payload(request: &HttpRequest, body: &[u8]) -> Result<ParsedSkillPayload, ApiError> {
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if content_type.starts_with("application/json") || looks_like_json(body) {
        let value: Value = serde_json::from_slice(body)
            .map_err(|err| ApiError::bad_request(format!("invalid skill JSON: {err}")))?;
        return parse_skill_json_payload(&value);
    }
    let upload = parse_file_upload(request, body)?;
    if upload.content.is_empty() {
        return Err(ApiError::bad_request("skill content must not be empty"));
    }
    let name = query_param(request.query_string(), "name")
        .or_else(|| Some(upload.filename.trim_end_matches(".md").to_string()))
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| ApiError::bad_request("skill requires name"))?;
    let description = query_param(request.query_string(), "description");
    Ok(ParsedSkillPayload {
        name,
        description,
        metadata: Value::Null,
        content_type: content_type_for_filename(&upload.filename).to_string(),
        content: upload.content,
    })
}

fn current_skill_version(state: &AppState, skill_id: &str) -> Result<SkillVersionRecord, ApiError> {
    let skills = lock_skills(state)?;
    let record = skills
        .get(skill_id)
        .ok_or_else(|| ApiError::not_found(format!("skill '{skill_id}' does not exist")))?;
    let version_id = record
        .current_version_id
        .as_deref()
        .ok_or_else(|| ApiError::not_found(format!("skill '{skill_id}' has no versions")))?;
    record.versions.get(version_id).cloned().ok_or_else(|| {
        ApiError::not_found(format!(
            "current version '{version_id}' for skill '{skill_id}' does not exist"
        ))
    })
}

fn skill_version(
    state: &AppState,
    skill_id: &str,
    version_id: &str,
) -> Result<SkillVersionRecord, ApiError> {
    let skills = lock_skills(state)?;
    let record = skills
        .get(skill_id)
        .ok_or_else(|| ApiError::not_found(format!("skill '{skill_id}' does not exist")))?;
    record.versions.get(version_id).cloned().ok_or_else(|| {
        ApiError::not_found(format!(
            "version '{version_id}' for skill '{skill_id}' does not exist"
        ))
    })
}

fn skill_json(record: &SkillRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "name": record.name,
        "description": record.description,
        "metadata": record.metadata,
        "status": record.status,
        "current_version_id": record.current_version_id,
        "version_count": record.versions.len()
    })
}

fn skill_version_json(record: &SkillVersionRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "skill_id": record.skill_id,
        "name": record.name,
        "description": record.description,
        "metadata": record.metadata,
        "status": record.status,
        "content_type": record.content_type,
        "bytes": record.content.len()
    })
}

fn lock_skills(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, SkillRecord>>, ApiError> {
    state
        .skills
        .lock()
        .map_err(|_| ApiError::internal("skill registry lock poisoned"))
}

fn optional_string_field(body: &Value, name: &'static str) -> Result<Option<String>, ApiError> {
    match body.get(name) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.clone())),
        Some(Value::String(_)) => Ok(None),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be a string"))),
    }
}

fn content_type_for_filename(filename: &str) -> &'static str {
    if filename.ends_with(".md") {
        "text/markdown; charset=utf-8"
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

fn looks_like_json(body: &[u8]) -> bool {
    body.iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace())
        .is_some_and(|byte| byte == b'{' || byte == b'[')
}

fn value_id(value: &Value) -> Option<Value> {
    value.get("id").cloned()
}
