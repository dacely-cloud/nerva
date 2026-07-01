use std::sync::atomic::Ordering;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{ApiError, AppState, authorize, query_param, unix_seconds};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OrganizationUsageKind {
    Completions,
    Embeddings,
    Moderations,
    Images,
    AudioSpeeches,
    AudioTranscriptions,
    VectorStores,
    CodeInterpreterSessions,
    FileSearchCalls,
    WebSearchCalls,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct OrganizationUsageTotals {
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) cached_tokens: u64,
    pub(crate) requests: u64,
    pub(crate) images: u64,
    pub(crate) audio_seconds: u64,
    pub(crate) audio_characters: u64,
    pub(crate) vector_store_bytes: u64,
    pub(crate) code_interpreter_sessions: u64,
    pub(crate) file_search_calls: u64,
    pub(crate) web_search_calls: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OrganizationUsageWindow {
    pub(crate) start_time: u64,
    pub(crate) end_time: u64,
    pub(crate) bucket_width: String,
}

pub(crate) async fn organization_usage_completions(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    organization_usage_response(state, request, OrganizationUsageKind::Completions).await
}

pub(crate) async fn organization_usage_embeddings(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    organization_usage_response(state, request, OrganizationUsageKind::Embeddings).await
}

pub(crate) async fn organization_usage_moderations(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    organization_usage_response(state, request, OrganizationUsageKind::Moderations).await
}

pub(crate) async fn organization_usage_images(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    organization_usage_response(state, request, OrganizationUsageKind::Images).await
}

pub(crate) async fn organization_usage_audio_speeches(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    organization_usage_response(state, request, OrganizationUsageKind::AudioSpeeches).await
}

pub(crate) async fn organization_usage_audio_transcriptions(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    organization_usage_response(state, request, OrganizationUsageKind::AudioTranscriptions).await
}

pub(crate) async fn organization_usage_vector_stores(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    organization_usage_response(state, request, OrganizationUsageKind::VectorStores).await
}

pub(crate) async fn organization_usage_code_interpreter_sessions(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    organization_usage_response(
        state,
        request,
        OrganizationUsageKind::CodeInterpreterSessions,
    )
    .await
}

pub(crate) async fn organization_usage_file_search_calls(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    organization_usage_response(state, request, OrganizationUsageKind::FileSearchCalls).await
}

pub(crate) async fn organization_usage_web_search_calls(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    organization_usage_response(state, request, OrganizationUsageKind::WebSearchCalls).await
}

pub(crate) async fn organization_costs(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let window = organization_query_window(request.query_string(), unix_seconds())?;
        let totals = organization_usage_totals(&state);
        Ok::<_, ApiError>(HttpResponse::Ok().json(organization_costs_page_json(&window, &totals)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

async fn organization_usage_response(
    state: web::Data<AppState>,
    request: HttpRequest,
    kind: OrganizationUsageKind,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let window = organization_query_window(request.query_string(), unix_seconds())?;
        let totals = organization_usage_totals(&state);
        Ok::<_, ApiError>(
            HttpResponse::Ok().json(organization_usage_page_json(kind, &window, &totals)),
        )
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn organization_usage_page_json(
    kind: OrganizationUsageKind,
    window: &OrganizationUsageWindow,
    totals: &OrganizationUsageTotals,
) -> Value {
    json!({
        "object": "page",
        "data": [{
            "object": "bucket",
            "start_time": window.start_time,
            "end_time": window.end_time,
            "results": [organization_usage_result_json(kind, totals)]
        }],
        "has_more": false,
        "next_page": null
    })
}

pub(crate) fn organization_costs_page_json(
    window: &OrganizationUsageWindow,
    totals: &OrganizationUsageTotals,
) -> Value {
    json!({
        "object": "page",
        "data": [{
            "object": "bucket",
            "start_time": window.start_time,
            "end_time": window.end_time,
            "results": [{
                "object": "organization.costs.result",
                "amount": {
                    "value": estimated_local_cost_usd(totals),
                    "currency": "usd"
                },
                "line_item": "nerva-local-estimate",
                "project_id": null
            }]
        }],
        "has_more": false,
        "next_page": null
    })
}

pub(crate) fn organization_query_window(
    query: &str,
    now: u64,
) -> Result<OrganizationUsageWindow, ApiError> {
    let bucket_width = query_param(query, "bucket_width").unwrap_or_else(|| "1d".to_string());
    if !matches!(bucket_width.as_str(), "1m" | "1h" | "1d") {
        return Err(ApiError::bad_request("bucket_width must be 1m, 1h, or 1d"));
    }
    if let Some(limit) = query_param(query, "limit") {
        let limit = limit
            .parse::<u64>()
            .map_err(|_| ApiError::bad_request("limit must be an integer"))?;
        if !(1..=180).contains(&limit) {
            return Err(ApiError::bad_request("limit must be between 1 and 180"));
        }
    }
    let fallback_width = bucket_seconds(&bucket_width);
    let start_time = query_u64(query, "start_time")?.unwrap_or(now.saturating_sub(fallback_width));
    let end_time = query_u64(query, "end_time")?.unwrap_or(now);
    if end_time <= start_time {
        return Err(ApiError::bad_request(
            "end_time must be greater than start_time",
        ));
    }
    Ok(OrganizationUsageWindow {
        start_time,
        end_time,
        bucket_width,
    })
}

fn organization_usage_result_json(
    kind: OrganizationUsageKind,
    totals: &OrganizationUsageTotals,
) -> Value {
    match kind {
        OrganizationUsageKind::Completions => json!({
            "object": kind.result_object(),
            "input_tokens": totals.input_tokens,
            "output_tokens": totals.output_tokens,
            "num_model_requests": totals.requests,
            "input_cached_tokens": totals.cached_tokens,
            "input_audio_tokens": 0,
            "output_audio_tokens": 0,
            "project_id": null,
            "user_id": null,
            "api_key_id": null,
            "model": null,
            "batch": null
        }),
        OrganizationUsageKind::Embeddings => json!({
            "object": kind.result_object(),
            "input_tokens": totals.input_tokens,
            "num_model_requests": totals.requests,
            "project_id": null,
            "user_id": null,
            "api_key_id": null,
            "model": null
        }),
        OrganizationUsageKind::Moderations => json!({
            "object": kind.result_object(),
            "input_tokens": totals.input_tokens,
            "num_model_requests": totals.requests,
            "project_id": null,
            "user_id": null,
            "api_key_id": null,
            "model": null
        }),
        OrganizationUsageKind::Images => json!({
            "object": kind.result_object(),
            "images": totals.images,
            "num_model_requests": totals.requests,
            "source": "image.generation",
            "size": null,
            "project_id": null,
            "user_id": null,
            "api_key_id": null,
            "model": null
        }),
        OrganizationUsageKind::AudioSpeeches => json!({
            "object": kind.result_object(),
            "characters": totals.audio_characters,
            "num_model_requests": totals.requests,
            "project_id": null,
            "user_id": null,
            "api_key_id": null,
            "model": null
        }),
        OrganizationUsageKind::AudioTranscriptions => json!({
            "object": kind.result_object(),
            "seconds": totals.audio_seconds,
            "num_model_requests": totals.requests,
            "project_id": null,
            "user_id": null,
            "api_key_id": null,
            "model": null
        }),
        OrganizationUsageKind::VectorStores => json!({
            "object": kind.result_object(),
            "usage_bytes": totals.vector_store_bytes,
            "project_id": null
        }),
        OrganizationUsageKind::CodeInterpreterSessions => json!({
            "object": kind.result_object(),
            "num_sessions": totals.code_interpreter_sessions,
            "project_id": null
        }),
        OrganizationUsageKind::FileSearchCalls => json!({
            "object": kind.result_object(),
            "num_calls": totals.file_search_calls,
            "project_id": null
        }),
        OrganizationUsageKind::WebSearchCalls => json!({
            "object": kind.result_object(),
            "num_calls": totals.web_search_calls,
            "project_id": null
        }),
    }
}

fn organization_usage_totals(state: &AppState) -> OrganizationUsageTotals {
    let mut totals = OrganizationUsageTotals {
        requests: state.request_count.load(Ordering::Relaxed),
        output_tokens: state.generated_tokens.load(Ordering::Relaxed),
        cached_tokens: state.scheduler_cache_hits.load(Ordering::Relaxed),
        ..OrganizationUsageTotals::default()
    };
    if let Ok(responses) = state.responses.lock() {
        for record in responses.values() {
            if let Some(usage) = record.response.get("usage") {
                totals.input_tokens = totals.input_tokens.saturating_add(
                    usage_u64(usage, "input_tokens")
                        .or_else(|| usage_u64(usage, "prompt_tokens"))
                        .unwrap_or(0),
                );
                totals.output_tokens = totals.output_tokens.max(
                    usage_u64(usage, "output_tokens")
                        .or_else(|| usage_u64(usage, "completion_tokens"))
                        .unwrap_or(0),
                );
            }
        }
    }
    if totals.input_tokens == 0
        && let Ok(sessions) = state.sessions.lock()
    {
        totals.input_tokens = sessions
            .values()
            .map(|session| session.prompt_tokens)
            .sum::<u64>();
    }
    if totals.requests == 0
        && let Ok(batches) = state.batches.lock()
    {
        totals.requests = batches
            .values()
            .map(|batch| batch.request_counts.total)
            .sum::<u64>();
    }
    if let Ok(videos) = state.videos.lock() {
        totals.images = totals.images.saturating_add(videos.len() as u64);
    }
    if let Ok(containers) = state.containers.lock() {
        totals.code_interpreter_sessions = containers.len() as u64;
    }
    if let Ok(vector_stores) = state.vector_stores.lock() {
        totals.vector_store_bytes = vector_stores
            .values()
            .flat_map(|store| store.files.values())
            .map(|file| file.usage_bytes as u64)
            .sum::<u64>();
        totals.file_search_calls = vector_stores
            .values()
            .map(|store| store.files.len() as u64)
            .sum::<u64>();
    }
    totals
}

fn usage_u64(usage: &Value, field: &str) -> Option<u64> {
    usage.get(field).and_then(Value::as_u64)
}

fn query_u64(query: &str, name: &str) -> Result<Option<u64>, ApiError> {
    query_param(query, name)
        .map(|value| {
            value
                .parse::<u64>()
                .map_err(|_| ApiError::bad_request(format!("{name} must be an integer")))
        })
        .transpose()
}

fn bucket_seconds(bucket_width: &str) -> u64 {
    match bucket_width {
        "1m" => 60,
        "1h" => 60 * 60,
        _ => 24 * 60 * 60,
    }
}

fn estimated_local_cost_usd(totals: &OrganizationUsageTotals) -> f64 {
    let token_units = totals.input_tokens.saturating_add(totals.output_tokens) as f64 / 1_000_000.0;
    let storage_units = totals.vector_store_bytes as f64 / 1_000_000_000.0;
    ((token_units * 0.25 + storage_units * 0.10) * 1_000_000.0).round() / 1_000_000.0
}

impl OrganizationUsageKind {
    fn result_object(self) -> &'static str {
        match self {
            OrganizationUsageKind::Completions => "organization.usage.completions.result",
            OrganizationUsageKind::Embeddings => "organization.usage.embeddings.result",
            OrganizationUsageKind::Moderations => "organization.usage.moderations.result",
            OrganizationUsageKind::Images => "organization.usage.images.result",
            OrganizationUsageKind::AudioSpeeches => "organization.usage.audio_speeches.result",
            OrganizationUsageKind::AudioTranscriptions => {
                "organization.usage.audio_transcriptions.result"
            }
            OrganizationUsageKind::VectorStores => "organization.usage.vector_stores.result",
            OrganizationUsageKind::CodeInterpreterSessions => {
                "organization.usage.code_interpreter_sessions.result"
            }
            OrganizationUsageKind::FileSearchCalls => "organization.usage.file_search_calls.result",
            OrganizationUsageKind::WebSearchCalls => "organization.usage.web_search_calls.result",
        }
    }
}
