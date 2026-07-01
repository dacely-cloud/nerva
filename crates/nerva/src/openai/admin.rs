use std::sync::atomic::Ordering;

use actix_web::{HttpRequest, HttpResponse, web};
use nerva_core::types::id::token::TokenId;
use nerva_model::hf::tokenizer::{
    PromptFormat, decode_generated_text, encode_text_prompt, format_prompt_for_model,
};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, authorize, delete_fine_tuned_model_alias, list_model_values,
    model_record_value, require_known_model, string_any,
};

pub(crate) async fn health() -> HttpResponse {
    HttpResponse::Ok().body("OK")
}

pub(crate) async fn version() -> HttpResponse {
    HttpResponse::Ok().json(json!({
        "version": env!("CARGO_PKG_VERSION"),
        "engine": "nerva"
    }))
}

pub(crate) async fn metrics(state: web::Data<AppState>) -> HttpResponse {
    let body = format!(
        concat!(
            "# TYPE nerva_openai_requests_total counter\n",
            "nerva_openai_requests_total {}\n",
            "# TYPE nerva_openai_generated_tokens_total counter\n",
            "nerva_openai_generated_tokens_total {}\n",
            "# TYPE nerva_openai_scheduler_active gauge\n",
            "nerva_openai_scheduler_active {}\n",
            "# TYPE nerva_openai_scheduler_admitted_total counter\n",
            "nerva_openai_scheduler_admitted_total {}\n",
            "# TYPE nerva_openai_scheduler_completed_total counter\n",
            "nerva_openai_scheduler_completed_total {}\n",
            "# TYPE nerva_openai_context_cache_hits_total counter\n",
            "nerva_openai_context_cache_hits_total {}\n",
            "# TYPE nerva_openai_context_cache_misses_total counter\n",
            "nerva_openai_context_cache_misses_total {}\n"
        ),
        state.request_count.load(Ordering::Relaxed),
        state.generated_tokens.load(Ordering::Relaxed),
        state.scheduler_active.load(Ordering::Relaxed),
        state.scheduler_admitted.load(Ordering::Relaxed),
        state.scheduler_completed.load(Ordering::Relaxed),
        state.scheduler_cache_hits.load(Ordering::Relaxed),
        state.scheduler_cache_misses.load(Ordering::Relaxed)
    );
    HttpResponse::Ok()
        .content_type("text/plain; version=0.0.4")
        .body(body)
}

pub(crate) async fn models(state: web::Data<AppState>, request: HttpRequest) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    HttpResponse::Ok().json(json!({
        "object": "list",
        "data": list_model_values(&state)
    }))
}

pub(crate) async fn model(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    let requested = path.into_inner();
    match model_record_value(&state, &requested) {
        Some(model) => HttpResponse::Ok().json(model),
        None => ApiError::not_found(format!(
            "model '{requested}' is not served by this NERVA instance"
        ))
        .into_response(),
    }
}

pub(crate) async fn delete_model(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    let requested = path.into_inner();
    let deleted = delete_fine_tuned_model_alias(&state, &requested);
    HttpResponse::Ok().json(json!({
        "id": requested,
        "object": "model",
        "deleted": deleted
    }))
}

pub(crate) async fn tokenize(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        require_known_model(&state, &body)?;
        let prompt = string_any(&body, &["prompt", "text", "input"])?
            .ok_or_else(|| ApiError::bad_request("tokenize requires prompt, text, or input"))?;
        let formatted =
            format_prompt_for_model(&state.config.model_path, &prompt, PromptFormat::Raw)
                .map_err(ApiError::bad_request)?;
        let encoded = encode_text_prompt(&state.config.model_path, &formatted.text)
            .map_err(ApiError::bad_request)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "tokens": encoded.token_ids,
            "count": encoded.token_ids.len()
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn detokenize(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        require_known_model(&state, &body)?;
        let tokens = body
            .get("tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| ApiError::bad_request("detokenize requires tokens array"))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|token| u32::try_from(token).ok())
                    .map(TokenId)
                    .ok_or_else(|| ApiError::bad_request("tokens must be u32 token ids"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let text = decode_generated_text(&state.config.model_path, &tokens)
            .map_err(ApiError::bad_request)?
            .ok_or_else(|| ApiError::bad_request("tokenizer decode is unavailable"))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "prompt": text,
            "text": text
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn unsupported_pooling(
    request: HttpRequest,
    state: web::Data<AppState>,
) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    ApiError::unsupported("pooling, classify, score, and rerank require a pooling/ranking backend; this NERVA build serves causal LM text generation only").into_response()
}

pub(crate) async fn unsupported_lora(
    request: HttpRequest,
    state: web::Data<AppState>,
) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    ApiError::unsupported("LoRA adapter hot-load/unload is not implemented for the current NERVA CUDA generation path").into_response()
}

pub(crate) async fn unsupported_admin_state(
    request: HttpRequest,
    state: web::Data<AppState>,
) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    ApiError::unsupported("engine sleep, wake, prefix-cache reset, and profiler controls are not implemented for this NERVA server").into_response()
}

pub(crate) async fn is_sleeping(request: HttpRequest, state: web::Data<AppState>) -> HttpResponse {
    if let Err(error) = authorize(&state, &request) {
        return error.into_response();
    }
    HttpResponse::Ok().json(json!({"is_sleeping": false}))
}

pub(crate) async fn not_found(request: HttpRequest) -> HttpResponse {
    ApiError::not_found(format!("unknown route: {}", request.path())).into_response()
}
