use std::sync::atomic::Ordering;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{ApiError, AppState, ContextCacheEntry, ContextCacheState, authorize, unix_seconds};

pub(crate) async fn context_cache_status(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let cache = lock_context_cache(&state)?;
        let entries = cache
            .entries
            .values()
            .map(context_cache_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "context_cache",
            "entries": entries,
            "entry_count": cache.entries.len(),
            "hits": cache.hits,
            "misses": cache.misses
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn delete_context_cache(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let key = path.into_inner();
        let removed = lock_context_cache(&state)?.entries.remove(&key).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": key,
            "object": "context_cache.deleted",
            "deleted": removed
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn reset_context_cache(
    request: HttpRequest,
    state: web::Data<AppState>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let mut cache = lock_context_cache(&state)?;
        let removed = cache.entries.len();
        cache.entries.clear();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "context_cache.reset",
            "removed": removed
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn record_context_cache_probe(
    state: &AppState,
    key: &str,
    prompt_hash: u64,
    prompt_tokens: usize,
) -> Result<bool, ApiError> {
    let mut cache = lock_context_cache(state)?;
    let now = unix_seconds();
    let hit = match cache.entries.get_mut(key) {
        Some(entry) if entry.prompt_hash == prompt_hash => {
            entry.updated = now;
            entry.hits = entry.hits.saturating_add(1);
            true
        }
        Some(entry) => {
            entry.prompt_hash = prompt_hash;
            entry.prompt_tokens = prompt_tokens;
            entry.updated = now;
            entry.hits = 0;
            false
        }
        None => {
            cache.entries.insert(
                key.to_string(),
                ContextCacheEntry {
                    key: key.to_string(),
                    prompt_hash,
                    prompt_tokens,
                    created: now,
                    updated: now,
                    hits: 0,
                },
            );
            false
        }
    };
    if hit {
        cache.hits = cache.hits.saturating_add(1);
        state.scheduler_cache_hits.fetch_add(1, Ordering::Relaxed);
    } else {
        cache.misses = cache.misses.saturating_add(1);
        state.scheduler_cache_misses.fetch_add(1, Ordering::Relaxed);
    }
    Ok(hit)
}

fn lock_context_cache(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, ContextCacheState>, ApiError> {
    state
        .context_cache
        .lock()
        .map_err(|_| ApiError::internal("context cache lock poisoned"))
}

fn context_cache_json(entry: &ContextCacheEntry) -> Value {
    json!({
        "id": entry.key,
        "object": "context_cache_entry",
        "prompt_hash": format!("{:016x}", entry.prompt_hash),
        "prompt_tokens": entry.prompt_tokens,
        "created": entry.created,
        "updated": entry.updated,
        "hits": entry.hits
    })
}
