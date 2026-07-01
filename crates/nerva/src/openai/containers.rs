use std::collections::HashMap;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, ContainerFileRecord, ContainerRecord, authorize, lock_files,
    parse_file_upload, request_metadata, request_optional_string, unix_seconds,
};

pub(crate) async fn create_container(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let metadata = request_metadata(&body)?;
        let name = request_optional_string(&body, "name")?;
        let expires_after = body
            .get("expires_after")
            .cloned()
            .unwrap_or_else(|| json!({"anchor": "last_active_at", "days": 7}));
        let file_ids = request_file_ids(&body)?;
        let now = unix_seconds();
        let id = state.next_response_id("container");
        let mut record = ContainerRecord {
            id: id.clone(),
            object: "container",
            created_at: now,
            name,
            status: "running".to_string(),
            metadata,
            expires_after,
            last_active_at: Some(now),
            files: HashMap::new(),
        };
        copy_files_into_container(&state, &mut record, &file_ids)?;
        lock_containers(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(container_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_containers(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let data = lock_containers(&state)?
            .values()
            .map(container_json)
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

pub(crate) async fn get_container(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let containers = lock_containers(&state)?;
        let record = containers
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("container '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(container_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_container(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let deleted = lock_containers(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "container.deleted",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_container_file(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Bytes,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let container_id = path.into_inner();
        let mut containers = lock_containers(&state)?;
        let container = containers.get_mut(&container_id).ok_or_else(|| {
            ApiError::not_found(format!("container '{container_id}' does not exist"))
        })?;
        let file = if request_json_file_id(&request, &body) {
            let value: Value = serde_json::from_slice(&body).map_err(|err| {
                ApiError::bad_request(format!("invalid container file JSON: {err}"))
            })?;
            copy_file_from_request(&state, container, &value)?
        } else {
            let upload = parse_file_upload(&request, &body)?;
            if upload.content.is_empty() {
                return Err(ApiError::bad_request(
                    "container file content must not be empty",
                ));
            }
            let filename = upload.filename.clone();
            let file = ContainerFileRecord {
                id: state.next_response_id("cfile"),
                object: "container.file",
                created_at: unix_seconds(),
                container_id: container_id.clone(),
                path: container_file_path(&json!({"filename": filename}), &upload.filename),
                filename: upload.filename,
                bytes: upload.content.len(),
                source_file_id: None,
                content: upload.content,
            };
            container.files.insert(file.id.clone(), file.clone());
            file
        };
        container.last_active_at = Some(unix_seconds());
        Ok::<_, ApiError>(HttpResponse::Ok().json(container_file_json(&file)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_container_files(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let container_id = path.into_inner();
        let containers = lock_containers(&state)?;
        let container = containers.get(&container_id).ok_or_else(|| {
            ApiError::not_found(format!("container '{container_id}' does not exist"))
        })?;
        let data = container
            .files
            .values()
            .map(container_file_json)
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

pub(crate) async fn get_container_file(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (container_id, file_id) = path.into_inner();
        let file = container_file_record(&state, &container_id, &file_id)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(container_file_json(&file)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_container_file(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (container_id, file_id) = path.into_inner();
        let mut containers = lock_containers(&state)?;
        let container = containers.get_mut(&container_id).ok_or_else(|| {
            ApiError::not_found(format!("container '{container_id}' does not exist"))
        })?;
        let deleted = container.files.remove(&file_id).is_some();
        container.last_active_at = Some(unix_seconds());
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": file_id,
            "object": "container.file.deleted",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_container_file_content(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (container_id, file_id) = path.into_inner();
        let file = container_file_record(&state, &container_id, &file_id)?;
        Ok::<_, ApiError>(
            HttpResponse::Ok()
                .content_type(container_file_content_type(&file.filename))
                .body(file.content),
        )
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn container_file_path(body: &Value, filename: &str) -> String {
    let raw = body
        .get("path")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(filename);
    let normalized = raw.trim();
    if normalized.starts_with('/') {
        normalized.to_string()
    } else {
        format!("/mnt/data/{normalized}")
    }
}

pub(crate) fn container_file_content_type(filename: &str) -> &'static str {
    if filename.ends_with(".jsonl") {
        "application/jsonl"
    } else if filename.ends_with(".json") {
        "application/json"
    } else if filename.ends_with(".txt")
        || filename.ends_with(".py")
        || filename.ends_with(".md")
        || filename.ends_with(".csv")
    {
        "text/plain; charset=utf-8"
    } else {
        "application/octet-stream"
    }
}

fn copy_files_into_container(
    state: &AppState,
    container: &mut ContainerRecord,
    file_ids: &[String],
) -> Result<(), ApiError> {
    for file_id in file_ids {
        let value = json!({"file_id": file_id});
        let _ = copy_file_from_request(state, container, &value)?;
    }
    Ok(())
}

fn copy_file_from_request(
    state: &AppState,
    container: &mut ContainerRecord,
    body: &Value,
) -> Result<ContainerFileRecord, ApiError> {
    let file_id = body
        .get("file_id")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| ApiError::bad_request("container file copy requires file_id"))?;
    let files = lock_files(state)?;
    let source = files
        .get(file_id)
        .ok_or_else(|| ApiError::not_found(format!("file '{file_id}' does not exist")))?;
    let filename = body
        .get("filename")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(&source.filename)
        .to_string();
    let file = ContainerFileRecord {
        id: state.next_response_id("cfile"),
        object: "container.file",
        created_at: unix_seconds(),
        container_id: container.id.clone(),
        path: container_file_path(body, &filename),
        filename,
        bytes: source.content.len(),
        source_file_id: Some(source.id.clone()),
        content: source.content.clone(),
    };
    container.files.insert(file.id.clone(), file.clone());
    Ok(file)
}

fn request_file_ids(body: &Value) -> Result<Vec<String>, ApiError> {
    let Some(value) = body.get("file_ids") else {
        return Ok(Vec::new());
    };
    let Some(values) = value.as_array() else {
        return Err(ApiError::bad_request("file_ids must be an array"));
    };
    let mut ids = Vec::with_capacity(values.len());
    for value in values {
        let Some(id) = value.as_str().filter(|value| !value.trim().is_empty()) else {
            return Err(ApiError::bad_request(
                "file_ids must contain non-empty strings",
            ));
        };
        ids.push(id.to_string());
    }
    Ok(ids)
}

fn request_json_file_id(request: &HttpRequest, body: &[u8]) -> bool {
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if !(content_type.starts_with("application/json") || looks_like_json(body)) {
        return false;
    }
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| value.get("file_id").cloned())
        .is_some()
}

fn container_file_record(
    state: &AppState,
    container_id: &str,
    file_id: &str,
) -> Result<ContainerFileRecord, ApiError> {
    let containers = lock_containers(state)?;
    let container = containers
        .get(container_id)
        .ok_or_else(|| ApiError::not_found(format!("container '{container_id}' does not exist")))?;
    container.files.get(file_id).cloned().ok_or_else(|| {
        ApiError::not_found(format!(
            "file '{file_id}' does not exist in container '{container_id}'"
        ))
    })
}

fn container_json(record: &ContainerRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "name": record.name,
        "status": record.status,
        "metadata": record.metadata,
        "expires_after": record.expires_after,
        "last_active_at": record.last_active_at,
        "file_count": record.files.len()
    })
}

fn container_file_json(record: &ContainerFileRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "container_id": record.container_id,
        "filename": record.filename,
        "path": record.path,
        "bytes": record.bytes,
        "source_file_id": record.source_file_id
    })
}

fn lock_containers(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, ContainerRecord>>, ApiError> {
    state
        .containers
        .lock()
        .map_err(|_| ApiError::internal("container registry lock poisoned"))
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
