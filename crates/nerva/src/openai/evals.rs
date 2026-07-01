use std::collections::HashMap;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Map, Value, json};

use super::{
    ApiError, AppState, EvalRecord, EvalRunOutputItemRecord, EvalRunRecord, EvalRunResultCounts,
    authorize, lock_files, request_metadata, request_optional_string, unix_seconds,
};

pub(crate) async fn create_eval(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = state.next_response_id("eval");
        let now = unix_seconds();
        let record = EvalRecord {
            id: id.clone(),
            object: "eval",
            created_at: now,
            updated_at: now,
            name: request_eval_name(&body, &id)?,
            data_source_config: request_eval_data_source_config(&body),
            testing_criteria: request_eval_testing_criteria(&body),
            metadata: request_metadata(&body)?,
            status: request_eval_status(&body, "active")?,
            runs: HashMap::new(),
        };
        lock_evals(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(eval_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_evals(state: web::Data<AppState>, request: HttpRequest) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let data = lock_evals(&state)?
            .values()
            .map(eval_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(list_json(data)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_eval(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let evals = lock_evals(&state)?;
        let record = evals
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("eval '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(eval_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn update_eval(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let name = request_optional_string(&body, "name")?;
        let metadata = request_metadata(&body)?;
        let status = request_optional_string(&body, "status")?;
        let mut evals = lock_evals(&state)?;
        let record = evals
            .get_mut(&id)
            .ok_or_else(|| ApiError::not_found(format!("eval '{id}' does not exist")))?;
        if let Some(name) = name {
            record.name = name;
        }
        if let Some(data_source_config) = request_optional_config(&body, "data_source_config") {
            record.data_source_config = data_source_config;
        }
        if let Some(testing_criteria) = request_optional_config(&body, "testing_criteria") {
            record.testing_criteria = testing_criteria;
        }
        if body.get("metadata").is_some() {
            record.metadata = metadata;
        }
        if let Some(status) = status {
            record.status = status;
        }
        record.updated_at = unix_seconds();
        Ok::<_, ApiError>(HttpResponse::Ok().json(eval_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_eval(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let deleted = lock_evals(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "eval.deleted",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_eval_run(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let eval_id = path.into_inner();
        let metadata = request_metadata(&body)?;
        let requested_status = request_optional_string(&body, "status")?;
        let data_source_override = request_eval_run_data_source(&body);
        let data_source = {
            let evals = lock_evals(&state)?;
            let record = evals
                .get(&eval_id)
                .ok_or_else(|| ApiError::not_found(format!("eval '{eval_id}' does not exist")))?;
            data_source_override.unwrap_or_else(|| record.data_source_config.clone())
        };
        let run_id = state.next_response_id("evalrun");
        let output_items = eval_output_items_from_source(&state, &eval_id, &run_id, &data_source)?;
        let result_counts = eval_run_result_counts(output_items.iter());
        let status = requested_status.unwrap_or_else(|| "completed".to_string());
        let run = EvalRunRecord {
            id: run_id.clone(),
            object: "eval.run",
            created_at: unix_seconds(),
            eval_id: eval_id.clone(),
            status,
            data_source,
            metadata,
            result_counts,
            output_items: output_items
                .into_iter()
                .map(|item| (item.id.clone(), item))
                .collect(),
        };
        let mut evals = lock_evals(&state)?;
        let eval = evals
            .get_mut(&eval_id)
            .ok_or_else(|| ApiError::not_found(format!("eval '{eval_id}' does not exist")))?;
        eval.updated_at = unix_seconds();
        eval.runs.insert(run_id, run.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(eval_run_json(&run)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_eval_runs(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let eval_id = path.into_inner();
        let evals = lock_evals(&state)?;
        let record = evals
            .get(&eval_id)
            .ok_or_else(|| ApiError::not_found(format!("eval '{eval_id}' does not exist")))?;
        let data = record.runs.values().map(eval_run_json).collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(list_json(data)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_eval_run(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (eval_id, run_id) = path.into_inner();
        let evals = lock_evals(&state)?;
        let eval = evals
            .get(&eval_id)
            .ok_or_else(|| ApiError::not_found(format!("eval '{eval_id}' does not exist")))?;
        let run = eval.runs.get(&run_id).ok_or_else(|| {
            ApiError::not_found(format!(
                "eval run '{run_id}' does not exist for eval '{eval_id}'"
            ))
        })?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(eval_run_json(run)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn update_eval_run(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
    body: web::Bytes,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (eval_id, run_id) = path.into_inner();
        let body = parse_optional_json_body(&body)?;
        let metadata = request_metadata(&body)?;
        let status = request_optional_string(&body, "status")?;
        let should_cancel = status.as_deref() == Some("cancelled")
            || body
                .get("action")
                .and_then(Value::as_str)
                .is_some_and(|action| matches!(action, "cancel" | "cancelled"));
        let mut evals = lock_evals(&state)?;
        let eval = evals
            .get_mut(&eval_id)
            .ok_or_else(|| ApiError::not_found(format!("eval '{eval_id}' does not exist")))?;
        let run = eval.runs.get_mut(&run_id).ok_or_else(|| {
            ApiError::not_found(format!(
                "eval run '{run_id}' does not exist for eval '{eval_id}'"
            ))
        })?;
        if body.get("metadata").is_some() {
            run.metadata = metadata;
        }
        if should_cancel {
            run.status = "cancelled".to_string();
        } else if let Some(status) = status {
            run.status = status;
        }
        eval.updated_at = unix_seconds();
        Ok::<_, ApiError>(HttpResponse::Ok().json(eval_run_json(run)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn cancel_eval_run(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (eval_id, run_id) = path.into_inner();
        let mut evals = lock_evals(&state)?;
        let eval = evals
            .get_mut(&eval_id)
            .ok_or_else(|| ApiError::not_found(format!("eval '{eval_id}' does not exist")))?;
        let run = eval.runs.get_mut(&run_id).ok_or_else(|| {
            ApiError::not_found(format!(
                "eval run '{run_id}' does not exist for eval '{eval_id}'"
            ))
        })?;
        run.status = "cancelled".to_string();
        eval.updated_at = unix_seconds();
        Ok::<_, ApiError>(HttpResponse::Ok().json(eval_run_json(run)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_eval_run(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (eval_id, run_id) = path.into_inner();
        let mut evals = lock_evals(&state)?;
        let eval = evals
            .get_mut(&eval_id)
            .ok_or_else(|| ApiError::not_found(format!("eval '{eval_id}' does not exist")))?;
        let deleted = eval.runs.remove(&run_id).is_some();
        eval.updated_at = unix_seconds();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": run_id,
            "object": "eval.run.deleted",
            "deleted": deleted
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_eval_run_output_items(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (eval_id, run_id) = path.into_inner();
        let evals = lock_evals(&state)?;
        let eval = evals
            .get(&eval_id)
            .ok_or_else(|| ApiError::not_found(format!("eval '{eval_id}' does not exist")))?;
        let run = eval.runs.get(&run_id).ok_or_else(|| {
            ApiError::not_found(format!(
                "eval run '{run_id}' does not exist for eval '{eval_id}'"
            ))
        })?;
        let data = run
            .output_items
            .values()
            .map(eval_output_item_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(list_json(data)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_eval_run_output_item(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (eval_id, run_id, output_item_id) = path.into_inner();
        let evals = lock_evals(&state)?;
        let eval = evals
            .get(&eval_id)
            .ok_or_else(|| ApiError::not_found(format!("eval '{eval_id}' does not exist")))?;
        let run = eval.runs.get(&run_id).ok_or_else(|| {
            ApiError::not_found(format!(
                "eval run '{run_id}' does not exist for eval '{eval_id}'"
            ))
        })?;
        let item = run.output_items.get(&output_item_id).ok_or_else(|| {
            ApiError::not_found(format!(
                "eval run output item '{output_item_id}' does not exist"
            ))
        })?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(eval_output_item_json(item)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn eval_output_items_from_inline_content(
    eval_id: &str,
    run_id: &str,
    content: &Value,
) -> Vec<EvalRunOutputItemRecord> {
    match content {
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(index, value)| eval_output_item_record(eval_id, run_id, index, value))
            .collect(),
        Value::Null => Vec::new(),
        value => vec![eval_output_item_record(eval_id, run_id, 0, value)],
    }
}

pub(crate) fn eval_run_result_counts<'a>(
    items: impl IntoIterator<Item = &'a EvalRunOutputItemRecord>,
) -> EvalRunResultCounts {
    let mut counts = EvalRunResultCounts::default();
    for item in items {
        counts.total = counts.total.saturating_add(1);
        match item.status.as_str() {
            "error" | "errored" => counts.errored = counts.errored.saturating_add(1),
            "fail" | "failed" => counts.failed = counts.failed.saturating_add(1),
            "cancelled" | "canceled" => {}
            _ => counts.passed = counts.passed.saturating_add(1),
        }
    }
    counts
}

fn eval_output_items_from_source(
    state: &AppState,
    eval_id: &str,
    run_id: &str,
    source: &Value,
) -> Result<Vec<EvalRunOutputItemRecord>, ApiError> {
    if let Some(content) = source_inline_content(source) {
        return Ok(eval_output_items_from_inline_content(
            eval_id, run_id, content,
        ));
    }
    if let Some(file_id) = source_file_id(source) {
        let files = lock_files(state)?;
        let file = files
            .get(&file_id)
            .ok_or_else(|| ApiError::not_found(format!("file '{file_id}' does not exist")))?;
        return Ok(eval_output_items_from_file_content(
            eval_id,
            run_id,
            &file.content,
        ));
    }
    if source.is_null()
        || source
            .as_object()
            .is_some_and(|object| object.is_empty() || object.get("type") == Some(&json!("custom")))
    {
        return Ok(Vec::new());
    }
    Ok(eval_output_items_from_inline_content(
        eval_id, run_id, source,
    ))
}

fn eval_output_items_from_file_content(
    eval_id: &str,
    run_id: &str,
    content: &[u8],
) -> Vec<EvalRunOutputItemRecord> {
    let text = String::from_utf8_lossy(content);
    let trimmed = text.trim();
    if trimmed.starts_with('[') {
        if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
            return eval_output_items_from_inline_content(eval_id, run_id, &value);
        }
    }
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() {
                None
            } else {
                Some(serde_json::from_str::<Value>(line).unwrap_or_else(|_| json!({"text": line})))
            }
        })
        .enumerate()
        .map(|(index, value)| eval_output_item_record(eval_id, run_id, index, &value))
        .collect()
}

fn eval_output_item_record(
    eval_id: &str,
    run_id: &str,
    index: usize,
    value: &Value,
) -> EvalRunOutputItemRecord {
    let object = value.as_object();
    let results = object
        .and_then(|object| object.get("results"))
        .map(results_value)
        .unwrap_or_default();
    EvalRunOutputItemRecord {
        id: format!("evalout-{run_id}-{index:x}"),
        object: "eval.run.output_item",
        created_at: unix_seconds(),
        eval_id: eval_id.to_string(),
        run_id: run_id.to_string(),
        status: output_item_status(object),
        item: output_item_value(object, value),
        sample: output_item_sample(object),
        results,
    }
}

fn output_item_value(object: Option<&Map<String, Value>>, fallback: &Value) -> Value {
    object
        .and_then(|object| {
            object
                .get("item")
                .or_else(|| object.get("input"))
                .or_else(|| object.get("messages"))
                .or_else(|| object.get("prompt"))
                .or_else(|| object.get("request"))
        })
        .cloned()
        .unwrap_or_else(|| fallback.clone())
}

fn output_item_sample(object: Option<&Map<String, Value>>) -> Value {
    object
        .and_then(|object| {
            object
                .get("sample")
                .or_else(|| object.get("expected_output"))
                .or_else(|| object.get("ideal"))
                .or_else(|| object.get("target"))
                .or_else(|| object.get("completion"))
                .or_else(|| object.get("response"))
        })
        .cloned()
        .unwrap_or(Value::Null)
}

fn output_item_status(object: Option<&Map<String, Value>>) -> String {
    if object
        .and_then(|object| object.get("passed"))
        .and_then(Value::as_bool)
        == Some(false)
    {
        return "failed".to_string();
    }
    object
        .and_then(|object| object.get("status"))
        .and_then(Value::as_str)
        .filter(|status| !status.trim().is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| "passed".to_string())
}

fn results_value(value: &Value) -> Vec<Value> {
    match value {
        Value::Array(items) => items.clone(),
        Value::Null => Vec::new(),
        value => vec![value.clone()],
    }
}

fn source_inline_content(source: &Value) -> Option<&Value> {
    match source {
        Value::Array(_) => Some(source),
        Value::Object(object) => object
            .get("content")
            .or_else(|| object.get("data"))
            .or_else(|| object.get("rows"))
            .or_else(|| object.get("items"))
            .or_else(|| object.get("examples")),
        _ => None,
    }
}

fn source_file_id(source: &Value) -> Option<String> {
    let object = source.as_object()?;
    for key in ["file_id", "input_file_id"] {
        if let Some(id) = object
            .get(key)
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
        {
            return Some(id.to_string());
        }
    }
    let source_type = object.get("type").and_then(Value::as_str);
    object
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| {
            matches!(source_type, Some("file_id" | "file" | "jsonl"))
                || id.starts_with("file-")
                || id.starts_with("file_")
        })
        .map(str::to_string)
}

fn request_eval_name(body: &Value, default: &str) -> Result<String, ApiError> {
    match body.get("name") {
        Some(Value::String(name)) if !name.trim().is_empty() => Ok(name.clone()),
        Some(Value::String(_)) => Err(ApiError::bad_request("eval name must not be empty")),
        Some(Value::Null) | None => Ok(default.to_string()),
        Some(_) => Err(ApiError::bad_request("eval name must be a string")),
    }
}

fn request_eval_data_source_config(body: &Value) -> Value {
    body.get("data_source_config")
        .or_else(|| body.get("data_source"))
        .or_else(|| body.get("source"))
        .cloned()
        .unwrap_or_else(|| json!({"type": "custom"}))
}

fn request_eval_testing_criteria(body: &Value) -> Value {
    body.get("testing_criteria")
        .or_else(|| body.get("criteria"))
        .cloned()
        .unwrap_or_else(|| json!([]))
}

fn request_eval_run_data_source(body: &Value) -> Option<Value> {
    body.get("data_source")
        .or_else(|| body.get("source"))
        .cloned()
}

fn request_eval_status(body: &Value, default: &str) -> Result<String, ApiError> {
    match body.get("status") {
        Some(Value::String(status)) if !status.trim().is_empty() => Ok(status.clone()),
        Some(Value::String(_)) => Err(ApiError::bad_request("status must not be empty")),
        Some(Value::Null) | None => Ok(default.to_string()),
        Some(_) => Err(ApiError::bad_request("status must be a string")),
    }
}

fn request_optional_config(body: &Value, field: &'static str) -> Option<Value> {
    body.get(field).filter(|value| !value.is_null()).cloned()
}

fn parse_optional_json_body(body: &[u8]) -> Result<Value, ApiError> {
    if body.iter().all(|byte| byte.is_ascii_whitespace()) {
        return Ok(json!({}));
    }
    serde_json::from_slice(body)
        .map_err(|err| ApiError::bad_request(format!("invalid eval run JSON body: {err}")))
}

fn eval_json(record: &EvalRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "name": record.name,
        "data_source_config": record.data_source_config,
        "testing_criteria": record.testing_criteria,
        "metadata": record.metadata,
        "status": record.status
    })
}

fn eval_run_json(record: &EvalRunRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "eval_id": record.eval_id,
        "status": record.status,
        "data_source": record.data_source,
        "metadata": record.metadata,
        "result_counts": eval_run_result_counts_json(record.result_counts)
    })
}

fn eval_output_item_json(record: &EvalRunOutputItemRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "eval_id": record.eval_id,
        "run_id": record.run_id,
        "status": record.status,
        "item": record.item,
        "sample": record.sample,
        "results": record.results
    })
}

fn eval_run_result_counts_json(counts: EvalRunResultCounts) -> Value {
    json!({
        "total": counts.total,
        "errored": counts.errored,
        "failed": counts.failed,
        "passed": counts.passed
    })
}

fn list_json(data: Vec<Value>) -> Value {
    json!({
        "object": "list",
        "data": data,
        "first_id": data.first().and_then(value_id),
        "last_id": data.last().and_then(value_id),
        "has_more": false
    })
}

fn value_id(value: &Value) -> Option<Value> {
    value.get("id").cloned()
}

fn lock_evals(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, EvalRecord>>, ApiError> {
    state
        .evals
        .lock()
        .map_err(|_| ApiError::internal("eval registry lock poisoned"))
}
