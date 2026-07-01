use std::collections::HashMap;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, FineTuningCheckpointPermissionRecord, FineTuningJobCheckpointRecord,
    FineTuningJobEventRecord, FineTuningJobRecord, authorize, lock_files, query_param,
    request_metadata, request_optional_string, request_u64_opt, unix_seconds,
};

pub(crate) async fn create_fine_tuning_job(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let training_file = required_string(&body, "training_file")?;
        let model = request_model(&state, &body)?;
        let validation_file = request_optional_string(&body, "validation_file")?;
        let training_bytes = file_size(&state, &training_file, "training_file")?;
        if let Some(validation_file) = validation_file.as_deref() {
            file_size(&state, validation_file, "validation_file")?;
        }
        let metadata = request_metadata(&body)?;
        let suffix = request_optional_string(&body, "suffix")?;
        let seed = request_u64_opt(&body, "seed")?;
        let hyperparameters = request_hyperparameters(&body);
        let method = request_method(&body, &hyperparameters);
        let integrations = body
            .get("integrations")
            .cloned()
            .unwrap_or_else(|| json!([]));
        let now = unix_seconds();
        let id = state.next_response_id("ftjob");
        let fine_tuned_model = fine_tuned_model_name(&model, suffix.as_deref(), &id);
        let checkpoint = FineTuningJobCheckpointRecord {
            id: state.next_response_id("ftckpt"),
            object: "fine_tuning.job.checkpoint",
            created_at: now,
            fine_tuned_model_checkpoint: format!("{fine_tuned_model}:ckpt-step-1"),
            fine_tuning_job_id: id.clone(),
            step_number: 1,
            metrics: json!({
                "step": 1,
                "train_loss": 0.0,
                "train_mean_token_accuracy": 1.0,
                "valid_loss": null,
                "valid_mean_token_accuracy": null,
                "full_valid_loss": null,
                "full_valid_mean_token_accuracy": null
            }),
            permissions: Vec::new(),
        };
        let events = vec![
            fine_tuning_event(&state, now, "info", "Fine-tuning job created", json!({})),
            fine_tuning_event(
                &state,
                now,
                "info",
                "Fine-tuning job completed locally using the served base model",
                json!({"fine_tuned_model": fine_tuned_model}),
            ),
        ];
        let record = FineTuningJobRecord {
            id: id.clone(),
            object: "fine_tuning.job",
            created_at: now,
            finished_at: Some(now),
            model,
            fine_tuned_model: Some(fine_tuned_model),
            training_file,
            validation_file,
            status: "succeeded".to_string(),
            hyperparameters,
            method,
            integrations,
            seed,
            suffix,
            trained_tokens: Some(approx_training_tokens(training_bytes)),
            estimated_finish: None,
            result_files: Vec::new(),
            metadata,
            error: Value::Null,
            events,
            checkpoints: vec![checkpoint],
        };
        lock_fine_tuning_jobs(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(fine_tuning_job_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_fine_tuning_jobs(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let data = lock_fine_tuning_jobs(&state)?
            .values()
            .map(fine_tuning_job_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(list_json(data)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_fine_tuning_job(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let jobs = lock_fine_tuning_jobs(&state)?;
        let record = jobs
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("fine-tuning job '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(fine_tuning_job_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_fine_tuning_events(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let jobs = lock_fine_tuning_jobs(&state)?;
        let record = jobs
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("fine-tuning job '{id}' does not exist")))?;
        let data = record
            .events
            .iter()
            .map(fine_tuning_event_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(list_json(data)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn cancel_fine_tuning_job(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    update_fine_tuning_job_status(
        state,
        request,
        path,
        "cancelled",
        "Fine-tuning job cancelled",
    )
    .await
}

pub(crate) async fn pause_fine_tuning_job(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    update_fine_tuning_job_status(state, request, path, "paused", "Fine-tuning job paused").await
}

pub(crate) async fn resume_fine_tuning_job(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    update_fine_tuning_job_status(state, request, path, "running", "Fine-tuning job resumed").await
}

pub(crate) async fn list_fine_tuning_checkpoints(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let jobs = lock_fine_tuning_jobs(&state)?;
        let record = jobs
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("fine-tuning job '{id}' does not exist")))?;
        let data = record
            .checkpoints
            .iter()
            .map(fine_tuning_checkpoint_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(list_json(data)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_fine_tuning_checkpoint_permissions(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let checkpoint_id = path.into_inner();
        let project_ids = request_project_ids(&body)?;
        let now = unix_seconds();
        let mut jobs = lock_fine_tuning_jobs(&state)?;
        let checkpoint = find_checkpoint_mut(&mut jobs, &checkpoint_id)?;
        let mut created = Vec::new();
        for project_id in project_ids {
            if let Some(existing) = checkpoint
                .permissions
                .iter()
                .find(|permission| permission.project_id == project_id)
                .cloned()
            {
                created.push(existing);
                continue;
            }
            let permission = FineTuningCheckpointPermissionRecord {
                id: state.next_response_id("cp"),
                object: "checkpoint.permission",
                created_at: now,
                project_id,
            };
            checkpoint.permissions.push(permission.clone());
            created.push(permission);
        }
        let records = created.iter().collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(checkpoint_permission_list_json(records, false)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_fine_tuning_checkpoint_permissions(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let checkpoint_id = path.into_inner();
        let query = checkpoint_permission_list_query(request.query_string())?;
        let jobs = lock_fine_tuning_jobs(&state)?;
        let checkpoint = find_checkpoint(&jobs, &checkpoint_id)?;
        let mut records = checkpoint
            .permissions
            .iter()
            .filter(|permission| {
                query
                    .project_id
                    .as_deref()
                    .is_none_or(|project_id| permission.project_id == project_id)
            })
            .collect::<Vec<_>>();
        records.sort_by(|left, right| {
            left.created_at
                .cmp(&right.created_at)
                .then_with(|| left.id.cmp(&right.id))
        });
        if query.order == "descending" {
            records.reverse();
        }
        if let Some(after) = query.after.as_deref() {
            if let Some(position) = records.iter().position(|record| record.id == after) {
                records = records.into_iter().skip(position + 1).collect();
            }
        }
        let has_more = records.len() > query.limit;
        records.truncate(query.limit);
        Ok::<_, ApiError>(
            HttpResponse::Ok().json(checkpoint_permission_list_json(records, has_more)),
        )
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_fine_tuning_checkpoint_permission(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (checkpoint_id, permission_id) = path.into_inner();
        let jobs = lock_fine_tuning_jobs(&state)?;
        let checkpoint = find_checkpoint(&jobs, &checkpoint_id)?;
        let permission = checkpoint
            .permissions
            .iter()
            .find(|permission| permission.id == permission_id)
            .ok_or_else(|| {
                ApiError::not_found(format!(
                    "checkpoint permission '{permission_id}' does not exist"
                ))
            })?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(checkpoint_permission_json(permission)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_fine_tuning_checkpoint_permission(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<(String, String)>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let (checkpoint_id, permission_id) = path.into_inner();
        let mut jobs = lock_fine_tuning_jobs(&state)?;
        let checkpoint = find_checkpoint_mut(&mut jobs, &checkpoint_id)?;
        let before = checkpoint.permissions.len();
        checkpoint
            .permissions
            .retain(|permission| permission.id != permission_id);
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": permission_id,
            "object": "checkpoint.permission",
            "deleted": checkpoint.permissions.len() != before
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn fine_tuned_model_name(model: &str, suffix: Option<&str>, job_id: &str) -> String {
    let mut clean_model = sanitize_model_part(model);
    if clean_model.is_empty() {
        clean_model = "model".to_string();
    }
    let suffix = suffix
        .map(sanitize_model_part)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "nerva".to_string());
    format!("ft:{clean_model}:{suffix}:{job_id}")
}

pub(crate) fn checkpoint_permission_json(record: &FineTuningCheckpointPermissionRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "project_id": record.project_id
    })
}

pub(crate) fn checkpoint_permission_list_json(
    records: Vec<&FineTuningCheckpointPermissionRecord>,
    has_more: bool,
) -> Value {
    let data = records
        .iter()
        .map(|record| checkpoint_permission_json(record))
        .collect::<Vec<_>>();
    json!({
        "object": "list",
        "data": data,
        "first_id": records.first().map(|record| record.id.as_str()),
        "last_id": records.last().map(|record| record.id.as_str()),
        "has_more": has_more
    })
}

pub(crate) fn request_project_ids(body: &Value) -> Result<Vec<String>, ApiError> {
    let Some(ids) = body.get("project_ids") else {
        return Err(ApiError::bad_request("project_ids is required"));
    };
    let Value::Array(ids) = ids else {
        return Err(ApiError::bad_request("project_ids must be an array"));
    };
    if ids.is_empty() {
        return Err(ApiError::bad_request("project_ids must not be empty"));
    }
    let mut out = Vec::with_capacity(ids.len());
    for value in ids {
        match value {
            Value::String(project_id) if !project_id.trim().is_empty() => {
                if !out.iter().any(|existing| existing == project_id) {
                    out.push(project_id.clone());
                }
            }
            Value::String(_) => {
                return Err(ApiError::bad_request(
                    "project_ids must not contain empty ids",
                ));
            }
            _ => {
                return Err(ApiError::bad_request(
                    "project_ids must contain only strings",
                ));
            }
        }
    }
    Ok(out)
}

async fn update_fine_tuning_job_status(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    status: &'static str,
    message: &'static str,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let now = unix_seconds();
        let event = fine_tuning_event(&state, now, "info", message, json!({"status": status}));
        let mut jobs = lock_fine_tuning_jobs(&state)?;
        let record = jobs
            .get_mut(&id)
            .ok_or_else(|| ApiError::not_found(format!("fine-tuning job '{id}' does not exist")))?;
        if !is_terminal_status(&record.status) || status == "running" {
            record.status = status.to_string();
            if status == "cancelled" {
                record.finished_at = Some(now);
                record.fine_tuned_model = None;
                record.checkpoints.clear();
            } else if status == "running" {
                record.finished_at = None;
            }
            record.events.push(event);
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(fine_tuning_job_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

fn required_string(body: &Value, field: &'static str) -> Result<String, ApiError> {
    match body.get(field) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(value.clone()),
        Some(Value::String(_)) => Err(ApiError::bad_request(format!("{field} must not be empty"))),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be a string"))),
        None => Err(ApiError::bad_request(format!("{field} is required"))),
    }
}

fn request_model(state: &AppState, body: &Value) -> Result<String, ApiError> {
    match body.get("model") {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(value.clone()),
        Some(Value::String(_)) => Err(ApiError::bad_request("model must not be empty")),
        Some(_) => Err(ApiError::bad_request("model must be a string")),
        None => Ok(state.config.model_id.clone()),
    }
}

fn file_size(state: &AppState, file_id: &str, field: &'static str) -> Result<usize, ApiError> {
    let files = lock_files(state)?;
    files
        .get(file_id)
        .map(|file| file.bytes)
        .ok_or_else(|| ApiError::not_found(format!("{field} file '{file_id}' does not exist")))
}

fn request_hyperparameters(body: &Value) -> Value {
    body.get("hyperparameters")
        .cloned()
        .or_else(|| {
            body.get("method")
                .and_then(|method| method.get("supervised"))
                .and_then(|supervised| supervised.get("hyperparameters"))
                .cloned()
        })
        .unwrap_or_else(|| {
            json!({
                "n_epochs": "auto",
                "batch_size": "auto",
                "learning_rate_multiplier": "auto"
            })
        })
}

fn request_method(body: &Value, hyperparameters: &Value) -> Value {
    body.get("method").cloned().unwrap_or_else(|| {
        json!({
            "type": "supervised",
            "supervised": {
                "hyperparameters": hyperparameters
            }
        })
    })
}

fn approx_training_tokens(bytes: usize) -> u64 {
    let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
    bytes.saturating_add(3) / 4
}

fn fine_tuning_event(
    state: &AppState,
    created_at: u64,
    level: &str,
    message: &str,
    data: Value,
) -> FineTuningJobEventRecord {
    FineTuningJobEventRecord {
        id: state.next_response_id("ftevent"),
        object: "fine_tuning.job.event",
        created_at,
        level: level.to_string(),
        message: message.to_string(),
        data,
        event_type: "message".to_string(),
    }
}

fn is_terminal_status(status: &str) -> bool {
    matches!(status, "succeeded" | "failed" | "cancelled")
}

fn sanitize_model_part(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn fine_tuning_job_json(record: &FineTuningJobRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "finished_at": record.finished_at,
        "model": record.model,
        "fine_tuned_model": record.fine_tuned_model,
        "organization_id": null,
        "training_file": record.training_file,
        "validation_file": record.validation_file,
        "result_files": record.result_files,
        "status": record.status,
        "trained_tokens": record.trained_tokens,
        "estimated_finish": record.estimated_finish,
        "hyperparameters": record.hyperparameters,
        "method": record.method,
        "integrations": record.integrations,
        "seed": record.seed,
        "suffix": record.suffix,
        "metadata": record.metadata,
        "error": record.error
    })
}

fn fine_tuning_event_json(record: &FineTuningJobEventRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "level": record.level,
        "message": record.message,
        "data": record.data,
        "type": record.event_type
    })
}

fn fine_tuning_checkpoint_json(record: &FineTuningJobCheckpointRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "fine_tuned_model_checkpoint": record.fine_tuned_model_checkpoint,
        "fine_tuning_job_id": record.fine_tuning_job_id,
        "step_number": record.step_number,
        "metrics": record.metrics,
        "permissions": checkpoint_permission_list_json(record.permissions.iter().collect(), false)
    })
}

#[derive(Clone, Debug)]
struct CheckpointPermissionListQuery {
    after: Option<String>,
    limit: usize,
    order: String,
    project_id: Option<String>,
}

fn checkpoint_permission_list_query(
    query: &str,
) -> Result<CheckpointPermissionListQuery, ApiError> {
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
    let order = query_param(query, "order").unwrap_or_else(|| "descending".to_string());
    if !matches!(order.as_str(), "ascending" | "descending") {
        return Err(ApiError::bad_request(
            "order must be ascending or descending",
        ));
    }
    Ok(CheckpointPermissionListQuery {
        after: query_param(query, "after").filter(|value| !value.trim().is_empty()),
        limit,
        order,
        project_id: query_param(query, "project_id").filter(|value| !value.trim().is_empty()),
    })
}

fn find_checkpoint<'a>(
    jobs: &'a HashMap<String, FineTuningJobRecord>,
    checkpoint_id: &str,
) -> Result<&'a FineTuningJobCheckpointRecord, ApiError> {
    jobs.values()
        .flat_map(|job| job.checkpoints.iter())
        .find(|checkpoint| checkpoint_matches(checkpoint, checkpoint_id))
        .ok_or_else(|| ApiError::not_found(format!("checkpoint '{checkpoint_id}' does not exist")))
}

fn find_checkpoint_mut<'a>(
    jobs: &'a mut HashMap<String, FineTuningJobRecord>,
    checkpoint_id: &str,
) -> Result<&'a mut FineTuningJobCheckpointRecord, ApiError> {
    jobs.values_mut()
        .flat_map(|job| job.checkpoints.iter_mut())
        .find(|checkpoint| checkpoint_matches(checkpoint, checkpoint_id))
        .ok_or_else(|| ApiError::not_found(format!("checkpoint '{checkpoint_id}' does not exist")))
}

fn checkpoint_matches(checkpoint: &FineTuningJobCheckpointRecord, checkpoint_id: &str) -> bool {
    checkpoint.id == checkpoint_id || checkpoint.fine_tuned_model_checkpoint == checkpoint_id
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

fn lock_fine_tuning_jobs(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, FineTuningJobRecord>>, ApiError> {
    state
        .fine_tuning_jobs
        .lock()
        .map_err(|_| ApiError::internal("fine-tuning job registry lock poisoned"))
}
