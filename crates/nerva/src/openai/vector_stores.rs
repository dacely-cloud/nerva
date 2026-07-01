use std::collections::HashMap;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, FileRecord, VectorStoreFileBatchRecord, VectorStoreFileCounts,
    VectorStoreFileRecord, VectorStoreRecord, authorize, lock_files, request_metadata,
    request_optional_string, unix_seconds,
};

pub(crate) async fn create_vector_store(
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
        let file_ids = request_file_ids(&body, false)?;
        let now = unix_seconds();
        let id = state.next_response_id("vs");
        let mut record = VectorStoreRecord {
            id: id.clone(),
            object: "vector_store",
            created_at: now,
            name,
            metadata,
            status: "completed".to_string(),
            expires_after,
            expires_at: None,
            last_active_at: Some(now),
            files: HashMap::new(),
            file_batches: HashMap::new(),
        };
        attach_files_to_store(&state, &mut record, &file_ids, Value::Null)?;
        lock_vector_stores(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(vector_store_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_vector_stores(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let data = lock_vector_stores(&state)?
            .values()
            .map(vector_store_json)
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

pub(crate) async fn get_vector_store(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let stores = lock_vector_stores(&state)?;
        let record = stores
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("vector store '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(vector_store_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn update_vector_store(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let metadata = request_metadata(&body)?;
        let name = request_optional_string(&body, "name")?;
        let mut stores = lock_vector_stores(&state)?;
        let record = stores
            .get_mut(&id)
            .ok_or_else(|| ApiError::not_found(format!("vector store '{id}' does not exist")))?;
        if body.get("name").is_some() {
            record.name = name;
        }
        if body.get("metadata").is_some() {
            record.metadata = metadata;
        }
        if let Some(expires_after) = body.get("expires_after") {
            record.expires_after = expires_after.clone();
        }
        record.last_active_at = Some(unix_seconds());
        Ok::<_, ApiError>(HttpResponse::Ok().json(vector_store_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_vector_store(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let deleted = lock_vector_stores(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "vector_store.deleted",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn search_vector_store(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let query = request_search_query(&body)?;
        let max_num_results = body
            .get("max_num_results")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(10)
            .clamp(1, 100);
        let store = {
            let mut stores = lock_vector_stores(&state)?;
            let record = stores.get_mut(&id).ok_or_else(|| {
                ApiError::not_found(format!("vector store '{id}' does not exist"))
            })?;
            record.last_active_at = Some(unix_seconds());
            record.clone()
        };
        let files = lock_files(&state)?;
        let data = vector_store_search_results(&store, &files, &query, max_num_results);
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "vector_store.search_results.page",
            "search_query": query,
            "data": data,
            "has_more": false,
            "next_page": null
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_vector_store_file(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let vector_store_id = path.into_inner();
        let file_id = body
            .get("file_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| ApiError::bad_request("vector store file requires file_id"))?
            .to_string();
        let attributes = request_attributes(&body)?;
        let mut stores = lock_vector_stores(&state)?;
        let record = stores.get_mut(&vector_store_id).ok_or_else(|| {
            ApiError::not_found(format!("vector store '{vector_store_id}' does not exist"))
        })?;
        attach_files_to_store(&state, record, &[file_id.clone()], attributes)?;
        let file = record
            .files
            .get(&file_id)
            .ok_or_else(|| ApiError::internal("attached vector store file is missing"))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(vector_store_file_json(file)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_vector_store_files(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let vector_store_id = path.into_inner();
        let stores = lock_vector_stores(&state)?;
        let record = stores.get(&vector_store_id).ok_or_else(|| {
            ApiError::not_found(format!("vector store '{vector_store_id}' does not exist"))
        })?;
        let data = record
            .files
            .values()
            .map(vector_store_file_json)
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

pub(crate) async fn get_vector_store_file(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (vector_store_id, file_id) = path.into_inner();
        let file = vector_store_file_record(&state, &vector_store_id, &file_id)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(vector_store_file_json(&file)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn update_vector_store_file(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (vector_store_id, file_id) = path.into_inner();
        let attributes = request_attributes(&body)?;
        let mut stores = lock_vector_stores(&state)?;
        let store = stores.get_mut(&vector_store_id).ok_or_else(|| {
            ApiError::not_found(format!("vector store '{vector_store_id}' does not exist"))
        })?;
        let file = store.files.get_mut(&file_id).ok_or_else(|| {
            ApiError::not_found(format!(
                "file '{file_id}' is not attached to vector store '{vector_store_id}'"
            ))
        })?;
        file.attributes = attributes;
        store.last_active_at = Some(unix_seconds());
        Ok::<_, ApiError>(HttpResponse::Ok().json(vector_store_file_json(file)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_vector_store_file(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (vector_store_id, file_id) = path.into_inner();
        let mut stores = lock_vector_stores(&state)?;
        let store = stores.get_mut(&vector_store_id).ok_or_else(|| {
            ApiError::not_found(format!("vector store '{vector_store_id}' does not exist"))
        })?;
        let deleted = store.files.remove(&file_id).is_some();
        store.last_active_at = Some(unix_seconds());
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": file_id,
            "object": "vector_store.file.deleted",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_vector_store_file_content(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (vector_store_id, file_id) = path.into_inner();
        let _ = vector_store_file_record(&state, &vector_store_id, &file_id)?;
        let files = lock_files(&state)?;
        let file = files
            .get(&file_id)
            .ok_or_else(|| ApiError::not_found(format!("file '{file_id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "list",
            "data": [file_content_json(file)]
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_vector_store_file_batch(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let vector_store_id = path.into_inner();
        let file_ids = request_file_ids(&body, true)?;
        let attributes = request_attributes(&body)?;
        let mut stores = lock_vector_stores(&state)?;
        let store = stores.get_mut(&vector_store_id).ok_or_else(|| {
            ApiError::not_found(format!("vector store '{vector_store_id}' does not exist"))
        })?;
        attach_files_to_store(&state, store, &file_ids, attributes)?;
        let batch = VectorStoreFileBatchRecord {
            id: state.next_response_id("vsfb"),
            object: "vector_store.file_batch",
            created_at: unix_seconds(),
            vector_store_id: vector_store_id.clone(),
            status: "completed".to_string(),
            file_ids,
        };
        store.file_batches.insert(batch.id.clone(), batch.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(vector_store_file_batch_json(store, &batch)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_vector_store_file_batch(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (vector_store_id, batch_id) = path.into_inner();
        let stores = lock_vector_stores(&state)?;
        let store = stores.get(&vector_store_id).ok_or_else(|| {
            ApiError::not_found(format!("vector store '{vector_store_id}' does not exist"))
        })?;
        let batch = store.file_batches.get(&batch_id).ok_or_else(|| {
            ApiError::not_found(format!(
                "vector store file batch '{batch_id}' does not exist"
            ))
        })?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(vector_store_file_batch_json(store, batch)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_vector_store_file_batch_files(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (vector_store_id, batch_id) = path.into_inner();
        let stores = lock_vector_stores(&state)?;
        let store = stores.get(&vector_store_id).ok_or_else(|| {
            ApiError::not_found(format!("vector store '{vector_store_id}' does not exist"))
        })?;
        let batch = store.file_batches.get(&batch_id).ok_or_else(|| {
            ApiError::not_found(format!(
                "vector store file batch '{batch_id}' does not exist"
            ))
        })?;
        let data = batch
            .file_ids
            .iter()
            .filter_map(|file_id| store.files.get(file_id))
            .map(vector_store_file_json)
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

pub(crate) async fn cancel_vector_store_file_batch(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (vector_store_id, batch_id) = path.into_inner();
        let mut stores = lock_vector_stores(&state)?;
        let store = stores.get_mut(&vector_store_id).ok_or_else(|| {
            ApiError::not_found(format!("vector store '{vector_store_id}' does not exist"))
        })?;
        let mut batch = store.file_batches.get(&batch_id).cloned().ok_or_else(|| {
            ApiError::not_found(format!(
                "vector store file batch '{batch_id}' does not exist"
            ))
        })?;
        batch.status = "cancelled".to_string();
        for file_id in &batch.file_ids {
            if let Some(file) = store.files.get_mut(file_id) {
                file.status = "cancelled".to_string();
            }
        }
        store.file_batches.insert(batch_id, batch.clone());
        store.last_active_at = Some(unix_seconds());
        Ok::<_, ApiError>(HttpResponse::Ok().json(vector_store_file_batch_json(store, &batch)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn vector_store_file_counts(record: &VectorStoreRecord) -> VectorStoreFileCounts {
    let mut counts = VectorStoreFileCounts::default();
    for file in record.files.values() {
        counts.total = counts.total.saturating_add(1);
        match file.status.as_str() {
            "in_progress" => counts.in_progress = counts.in_progress.saturating_add(1),
            "failed" => counts.failed = counts.failed.saturating_add(1),
            "cancelled" => counts.cancelled = counts.cancelled.saturating_add(1),
            _ => counts.completed = counts.completed.saturating_add(1),
        }
    }
    counts
}

pub(crate) fn lexical_match_score(query: &str, text: &str) -> f64 {
    let terms = query_terms(query);
    if terms.is_empty() {
        return 0.0;
    }
    let text_lower = text.to_ascii_lowercase();
    let mut score = 0.0;
    for term in &terms {
        if text_lower.contains(term) {
            score += 1.0;
        }
    }
    score / terms.len() as f64
}

fn attach_files_to_store(
    state: &AppState,
    store: &mut VectorStoreRecord,
    file_ids: &[String],
    attributes: Value,
) -> Result<(), ApiError> {
    if file_ids.is_empty() {
        return Ok(());
    }
    let files = lock_files(state)?;
    for file_id in file_ids {
        let file = files
            .get(file_id)
            .ok_or_else(|| ApiError::not_found(format!("file '{file_id}' does not exist")))?;
        store.files.insert(
            file_id.clone(),
            VectorStoreFileRecord {
                id: file_id.clone(),
                object: "vector_store.file",
                created_at: unix_seconds(),
                vector_store_id: store.id.clone(),
                status: "completed".to_string(),
                last_error: None,
                usage_bytes: file.bytes,
                attributes: attributes.clone(),
            },
        );
    }
    store.last_active_at = Some(unix_seconds());
    Ok(())
}

fn vector_store_search_results(
    store: &VectorStoreRecord,
    files: &HashMap<String, FileRecord>,
    query: &str,
    max_num_results: usize,
) -> Vec<Value> {
    let mut scored = Vec::new();
    for attached in store.files.values() {
        if attached.status != "completed" {
            continue;
        }
        let Some(file) = files.get(&attached.id) else {
            continue;
        };
        let text = String::from_utf8_lossy(&file.content);
        let score = lexical_match_score(query, &text);
        if score <= 0.0 {
            continue;
        }
        scored.push((score, file, attached, text.to_string()));
    }
    scored.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.id.cmp(&right.1.id))
    });
    scored
        .into_iter()
        .take(max_num_results)
        .map(|(score, file, attached, text)| {
            json!({
                "file_id": file.id,
                "filename": file.filename,
                "score": score,
                "attributes": attached.attributes,
                "content": [{
                    "type": "text",
                    "text": search_snippet(&text, query)
                }]
            })
        })
        .collect()
}

fn vector_store_file_record(
    state: &AppState,
    vector_store_id: &str,
    file_id: &str,
) -> Result<VectorStoreFileRecord, ApiError> {
    let stores = lock_vector_stores(state)?;
    let store = stores.get(vector_store_id).ok_or_else(|| {
        ApiError::not_found(format!("vector store '{vector_store_id}' does not exist"))
    })?;
    store.files.get(file_id).cloned().ok_or_else(|| {
        ApiError::not_found(format!(
            "file '{file_id}' is not attached to vector store '{vector_store_id}'"
        ))
    })
}

fn request_file_ids(body: &Value, required: bool) -> Result<Vec<String>, ApiError> {
    let Some(value) = body.get("file_ids") else {
        if required {
            return Err(ApiError::bad_request("file_ids is required"));
        }
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
    if required && ids.is_empty() {
        return Err(ApiError::bad_request("file_ids must not be empty"));
    }
    Ok(ids)
}

fn request_attributes(body: &Value) -> Result<Value, ApiError> {
    match body.get("attributes") {
        Some(Value::Object(_)) => Ok(body.get("attributes").cloned().unwrap_or(Value::Null)),
        Some(Value::Null) | None => Ok(Value::Null),
        Some(_) => Err(ApiError::bad_request("attributes must be an object")),
    }
}

fn request_search_query(body: &Value) -> Result<String, ApiError> {
    match body.get("query") {
        Some(Value::String(query)) if !query.trim().is_empty() => Ok(query.clone()),
        Some(Value::Array(items)) => {
            let mut parts = Vec::new();
            for item in items {
                let Some(text) = item.as_str().filter(|value| !value.trim().is_empty()) else {
                    return Err(ApiError::bad_request(
                        "search query array must contain non-empty strings",
                    ));
                };
                parts.push(text.to_string());
            }
            if parts.is_empty() {
                Err(ApiError::bad_request("search query must not be empty"))
            } else {
                Ok(parts.join(" "))
            }
        }
        Some(Value::String(_)) => Err(ApiError::bad_request("search query must not be empty")),
        Some(_) => Err(ApiError::bad_request(
            "search query must be a string or string array",
        )),
        None => Err(ApiError::bad_request("search requires query")),
    }
}

fn vector_store_json(record: &VectorStoreRecord) -> Value {
    let counts = vector_store_file_counts(record);
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "name": record.name,
        "metadata": record.metadata,
        "status": record.status,
        "usage_bytes": record.files.values().map(|file| file.usage_bytes).sum::<usize>(),
        "file_counts": file_counts_json(counts),
        "expires_after": record.expires_after,
        "expires_at": record.expires_at,
        "last_active_at": record.last_active_at
    })
}

fn vector_store_file_json(record: &VectorStoreFileRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "vector_store_id": record.vector_store_id,
        "status": record.status,
        "last_error": record.last_error,
        "usage_bytes": record.usage_bytes,
        "attributes": record.attributes
    })
}

fn vector_store_file_batch_json(
    store: &VectorStoreRecord,
    record: &VectorStoreFileBatchRecord,
) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "vector_store_id": record.vector_store_id,
        "status": record.status,
        "file_counts": file_counts_json(vector_store_file_counts(store))
    })
}

fn file_counts_json(counts: VectorStoreFileCounts) -> Value {
    json!({
        "in_progress": counts.in_progress,
        "completed": counts.completed,
        "failed": counts.failed,
        "cancelled": counts.cancelled,
        "total": counts.total
    })
}

fn file_content_json(file: &FileRecord) -> Value {
    json!({
        "type": "text",
        "text": String::from_utf8_lossy(&file.content).to_string()
    })
}

fn search_snippet(text: &str, query: &str) -> String {
    let terms = query_terms(query);
    let text_lower = text.to_ascii_lowercase();
    let start = terms
        .iter()
        .filter_map(|term| text_lower.find(term))
        .min()
        .unwrap_or(0);
    let snippet_start = start.saturating_sub(80);
    text.chars().skip(snippet_start).take(400).collect()
}

fn query_terms(query: &str) -> Vec<String> {
    query
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .map(str::trim)
        .filter(|term| !term.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

fn lock_vector_stores(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, VectorStoreRecord>>, ApiError> {
    state
        .vector_stores
        .lock()
        .map_err(|_| ApiError::internal("vector store registry lock poisoned"))
}

fn value_id(value: &Value) -> Option<Value> {
    value.get("id").cloned()
}
