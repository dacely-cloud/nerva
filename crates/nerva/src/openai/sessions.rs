use std::collections::HashMap;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{ApiError, AppState, SessionRecord, StreamRunStats, authorize, unix_seconds};

pub(crate) async fn create_session(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = body
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| state.next_response_id("sess"));
        let now = unix_seconds();
        let record = SessionRecord {
            id: id.clone(),
            object: "session",
            created: now,
            updated: now,
            request_count: 0,
            prompt_tokens: 0,
            generated_tokens: 0,
            last_cache_key: None,
            last_prompt_hash: None,
        };
        lock_sessions(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(session_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_sessions(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let sessions = lock_sessions(&state)?
            .values()
            .map(session_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "list",
            "data": sessions
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn get_session(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let sessions = lock_sessions(&state)?;
        let record = sessions
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("session '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(session_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_session(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let removed = lock_sessions(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "session.deleted",
            "deleted": removed
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn ensure_session_for_request(
    state: &AppState,
    session_id: Option<&str>,
) -> Result<Option<String>, ApiError> {
    let Some(session_id) = session_id else {
        return Ok(None);
    };
    let mut sessions = lock_sessions(state)?;
    if !sessions.contains_key(session_id) {
        let now = unix_seconds();
        sessions.insert(
            session_id.to_string(),
            SessionRecord {
                id: session_id.to_string(),
                object: "session",
                created: now,
                updated: now,
                request_count: 0,
                prompt_tokens: 0,
                generated_tokens: 0,
                last_cache_key: None,
                last_prompt_hash: None,
            },
        );
    }
    Ok(Some(session_id.to_string()))
}

pub(crate) fn record_session_generation(state: &AppState, stats: &StreamRunStats) {
    let Some(session_id) = stats.session_id.as_deref() else {
        return;
    };
    let Ok(mut sessions) = state.sessions.lock() else {
        return;
    };
    if let Some(session) = sessions.get_mut(session_id) {
        session.updated = unix_seconds();
        session.request_count = session.request_count.saturating_add(1);
        session.prompt_tokens = session
            .prompt_tokens
            .saturating_add(stats.prompt_tokens as u64);
        session.generated_tokens = session
            .generated_tokens
            .saturating_add(stats.generated_tokens as u64);
        session.last_cache_key = Some(stats.cache_key.clone());
        session.last_prompt_hash = Some(stats.prompt_hash);
        let _ = stats.cache_hit;
    }
}

fn lock_sessions(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, SessionRecord>>, ApiError> {
    state
        .sessions
        .lock()
        .map_err(|_| ApiError::internal("session registry lock poisoned"))
}

pub(crate) fn session_json(record: &SessionRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created": record.created,
        "updated": record.updated,
        "request_count": record.request_count,
        "prompt_tokens": record.prompt_tokens,
        "generated_tokens": record.generated_tokens,
        "last_cache_key": record.last_cache_key,
        "last_prompt_hash": record.last_prompt_hash.map(|hash| format!("{hash:016x}"))
    })
}
