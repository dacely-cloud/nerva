use std::collections::HashMap;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Map, Value, json};

use super::{
    ApiError, AppState, RealtimeCallRecord, RealtimeClientSecretRecord, RealtimeSessionRecord,
    authorize, unix_seconds,
};

const DEFAULT_REALTIME_MODEL: &str = "gpt-realtime";
const DEFAULT_TRANSLATION_MODEL: &str = "gpt-realtime-translate";
const DEFAULT_TRANSLATION_TRANSCRIPTION_MODEL: &str = "gpt-realtime-whisper";
const DEFAULT_TRANSCRIPTION_MODEL: &str = "gpt-4o-transcribe";
const DEFAULT_VOICE: &str = "alloy";
const DEFAULT_AUDIO_FORMAT: &str = "pcm16";
const DEFAULT_TRANSLATION_LANGUAGE: &str = "en";
const REALTIME_SECRET_TTL_SECONDS: u64 = 60;
const REALTIME_SESSION_TTL_SECONDS: u64 = 30 * 60;
const TRANSLATION_SECRET_TTL_SECONDS: u64 = 10 * 60;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RealtimeSessionKind {
    Realtime,
    Transcription,
    Translation,
}

pub(crate) async fn create_realtime_client_secret(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let session_body = body.get("session").unwrap_or(&body);
        let record = create_realtime_record(&state, RealtimeSessionKind::Realtime, session_body)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(realtime_session_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_realtime_session(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let record = create_realtime_record(&state, RealtimeSessionKind::Realtime, &body)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(realtime_session_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_realtime_transcription_session(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let record = create_realtime_record(&state, RealtimeSessionKind::Transcription, &body)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(realtime_session_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_realtime_translation_client_secret(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let session_body = body.get("session").unwrap_or(&body);
        let now = unix_seconds();
        let expires_at = translation_secret_expires_at(&body, now)?;
        let record = create_realtime_record_with_expiry(
            &state,
            RealtimeSessionKind::Translation,
            session_body,
            expires_at,
        )?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(realtime_translation_client_secret_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn accept_realtime_call(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let call_id = path.into_inner();
        let session = create_realtime_record(&state, RealtimeSessionKind::Realtime, &body)?;
        let now = unix_seconds();
        let mut calls = lock_realtime_calls(&state)?;
        let record = calls
            .entry(call_id.clone())
            .or_insert_with(|| RealtimeCallRecord {
                id: call_id,
                object: "realtime.call",
                created_at: now,
                updated_at: now,
                status: "incoming".to_string(),
                session_id: None,
                status_code: None,
                target_uri: None,
                config: Value::Null,
            });
        record.updated_at = now;
        record.status = "accepted".to_string();
        record.session_id = Some(session.id.clone());
        record.status_code = None;
        record.target_uri = None;
        record.config = realtime_session_json(&session);
        Ok::<_, ApiError>(HttpResponse::Ok().json(realtime_call_json(record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn reject_realtime_call(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let status_code = request_realtime_reject_status(&body)?;
        let record = upsert_realtime_call(
            &state,
            path.into_inner(),
            "rejected",
            None,
            Some(status_code),
            None,
            Value::Null,
        )?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(realtime_call_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn hangup_realtime_call(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let record = upsert_realtime_call(
            &state,
            path.into_inner(),
            "ended",
            None,
            None,
            None,
            Value::Null,
        )?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(realtime_call_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn refer_realtime_call(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let target_uri = request_realtime_refer_target(&body)?;
        let record = upsert_realtime_call(
            &state,
            path.into_inner(),
            "referred",
            None,
            None,
            Some(target_uri),
            Value::Null,
        )?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(realtime_call_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn create_realtime_record(
    state: &AppState,
    kind: RealtimeSessionKind,
    body: &Value,
) -> Result<RealtimeSessionRecord, ApiError> {
    let now = unix_seconds();
    let expires_at = request_u64_field(body, "expires_at")?
        .unwrap_or_else(|| now.saturating_add(REALTIME_SESSION_TTL_SECONDS));
    create_realtime_record_with_expiry(state, kind, body, expires_at)
}

fn create_realtime_record_with_expiry(
    state: &AppState,
    kind: RealtimeSessionKind,
    body: &Value,
    expires_at: u64,
) -> Result<RealtimeSessionRecord, ApiError> {
    let now = unix_seconds();
    let id =
        optional_nonempty_string(body, "id")?.unwrap_or_else(|| state.next_response_id("sess"));
    let secret = RealtimeClientSecretRecord {
        value: realtime_client_secret_value(&id, now),
        expires_at: match kind {
            RealtimeSessionKind::Translation => expires_at,
            _ => now.saturating_add(REALTIME_SECRET_TTL_SECONDS),
        },
    };
    let config =
        realtime_config_value(&state.config.model_id, kind, body, &id, expires_at, &secret)?;
    let record = RealtimeSessionRecord {
        id: id.clone(),
        object: kind.object(),
        kind: kind.kind_name(),
        created_at: now,
        expires_at,
        client_secret: secret,
        config,
    };
    lock_realtime_sessions(state)?.insert(id, record.clone());
    Ok(record)
}

pub(crate) fn realtime_call_json(record: &RealtimeCallRecord) -> Value {
    json!({
        "id": record.id,
        "object": record.object,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "status": record.status,
        "session_id": record.session_id,
        "status_code": record.status_code,
        "target_uri": record.target_uri,
        "session": if record.config.is_null() { Value::Null } else { record.config.clone() },
        "nerva": {
            "backend": "realtime-call-placeholder"
        }
    })
}

pub(crate) fn realtime_translation_client_secret_json(record: &RealtimeSessionRecord) -> Value {
    json!({
        "value": record.client_secret.value,
        "expires_at": record.client_secret.expires_at,
        "session": realtime_translation_session_json(record)
    })
}

pub(crate) fn realtime_translation_session_json(record: &RealtimeSessionRecord) -> Value {
    let mut value = record.config.clone();
    value["id"] = json!(record.id);
    value["type"] = json!("translation");
    value["expires_at"] = json!(record.expires_at);
    value
}

pub(crate) fn realtime_session_json(record: &RealtimeSessionRecord) -> Value {
    let mut value = record.config.clone();
    value["id"] = json!(record.id);
    value["object"] = json!(record.object);
    value["created_at"] = json!(record.created_at);
    value["expires_at"] = json!(record.expires_at);
    value["client_secret"] = json!({
        "value": record.client_secret.value,
        "expires_at": record.client_secret.expires_at
    });
    value["nerva"] = json!({
        "kind": record.kind,
        "backend": "realtime-session-placeholder"
    });
    value
}

pub(crate) fn request_realtime_reject_status(body: &Value) -> Result<u16, ApiError> {
    match body.get("status_code") {
        Some(Value::Number(number)) => {
            let Some(status) = number.as_u64() else {
                return Err(ApiError::bad_request("status_code must be an integer"));
            };
            let status = u16::try_from(status)
                .map_err(|_| ApiError::bad_request("status_code must be a SIP response code"))?;
            if (300..=699).contains(&status) {
                Ok(status)
            } else {
                Err(ApiError::bad_request(
                    "status_code must be between 300 and 699",
                ))
            }
        }
        Some(Value::Null) | None => Ok(603),
        Some(_) => Err(ApiError::bad_request("status_code must be an integer")),
    }
}

pub(crate) fn request_realtime_refer_target(body: &Value) -> Result<String, ApiError> {
    match body.get("target_uri") {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(value.clone()),
        Some(Value::String(_)) => Err(ApiError::bad_request("target_uri must not be empty")),
        Some(_) => Err(ApiError::bad_request("target_uri must be a string")),
        None => Err(ApiError::bad_request("target_uri is required")),
    }
}

pub(crate) fn realtime_config_value(
    served_model_id: &str,
    kind: RealtimeSessionKind,
    body: &Value,
    id: &str,
    expires_at: u64,
    secret: &RealtimeClientSecretRecord,
) -> Result<Value, ApiError> {
    if !matches!(body, Value::Object(_)) {
        return Err(ApiError::bad_request(
            "realtime session request must be an object",
        ));
    }
    let mut config = body.as_object().cloned().unwrap_or_default();
    config.insert("id".to_string(), json!(id));
    config.insert("object".to_string(), json!(kind.object()));
    config.insert("expires_at".to_string(), json!(expires_at));
    config.insert(
        "client_secret".to_string(),
        json!({
            "value": secret.value,
            "expires_at": secret.expires_at
        }),
    );
    match kind {
        RealtimeSessionKind::Realtime => normalize_realtime_config(served_model_id, &mut config)?,
        RealtimeSessionKind::Transcription => normalize_transcription_config(&mut config)?,
        RealtimeSessionKind::Translation => normalize_translation_config(&mut config)?,
    }
    Ok(Value::Object(config))
}

fn normalize_realtime_config(
    served_model_id: &str,
    config: &mut Map<String, Value>,
) -> Result<(), ApiError> {
    if !config.contains_key("type") {
        config.insert("type".to_string(), json!("realtime"));
    }
    let model = config
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.trim().is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| {
            if served_model_id.starts_with("gpt-realtime") || served_model_id.contains("realtime") {
                served_model_id.to_string()
            } else {
                DEFAULT_REALTIME_MODEL.to_string()
            }
        });
    config.insert("model".to_string(), json!(model));
    if !config.contains_key("output_modalities") && !config.contains_key("modalities") {
        config.insert("output_modalities".to_string(), json!(["audio"]));
    }
    if !config.contains_key("modalities") {
        config.insert("modalities".to_string(), json!(["audio", "text"]));
    }
    if !config.contains_key("instructions") {
        config.insert(
            "instructions".to_string(),
            json!("You are a realtime NERVA session. Respond using the requested modalities."),
        );
    }
    if !config.contains_key("voice") {
        config.insert("voice".to_string(), json!(DEFAULT_VOICE));
    }
    if !config.contains_key("input_audio_format") {
        config.insert(
            "input_audio_format".to_string(),
            json!(DEFAULT_AUDIO_FORMAT),
        );
    }
    if !config.contains_key("output_audio_format") {
        config.insert(
            "output_audio_format".to_string(),
            json!(DEFAULT_AUDIO_FORMAT),
        );
    }
    if !config.contains_key("max_output_tokens")
        && !config.contains_key("max_response_output_tokens")
    {
        config.insert("max_output_tokens".to_string(), json!("inf"));
        config.insert("max_response_output_tokens".to_string(), json!("inf"));
    }
    if !config.contains_key("turn_detection") {
        config.insert(
            "turn_detection".to_string(),
            json!({
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
                "create_response": true,
                "interrupt_response": true
            }),
        );
    }
    if !config.contains_key("tools") {
        config.insert("tools".to_string(), json!([]));
    }
    if !config.contains_key("tool_choice") {
        config.insert("tool_choice".to_string(), json!("auto"));
    }
    if !config.contains_key("audio") {
        let input_transcription = config
            .get("input_audio_transcription")
            .cloned()
            .unwrap_or(Value::Null);
        let turn_detection = config.get("turn_detection").cloned().unwrap_or(Value::Null);
        config.insert(
            "audio".to_string(),
            json!({
                "input": {
                    "format": config.get("input_audio_format").cloned().unwrap_or_else(|| json!(DEFAULT_AUDIO_FORMAT)),
                    "transcription": input_transcription,
                    "turn_detection": turn_detection
                },
                "output": {
                    "format": config.get("output_audio_format").cloned().unwrap_or_else(|| json!(DEFAULT_AUDIO_FORMAT)),
                    "voice": config.get("voice").cloned().unwrap_or_else(|| json!(DEFAULT_VOICE))
                }
            }),
        );
    }
    validate_realtime_config(config)
}

fn normalize_transcription_config(config: &mut Map<String, Value>) -> Result<(), ApiError> {
    if !config.contains_key("type") {
        config.insert("type".to_string(), json!("realtime.transcription_session"));
    }
    if !config.contains_key("input_audio_format") {
        config.insert(
            "input_audio_format".to_string(),
            json!(DEFAULT_AUDIO_FORMAT),
        );
    }
    if !config.contains_key("input_audio_transcription") {
        config.insert(
            "input_audio_transcription".to_string(),
            json!({
                "model": DEFAULT_TRANSCRIPTION_MODEL
            }),
        );
    }
    if !config.contains_key("turn_detection") {
        config.insert(
            "turn_detection".to_string(),
            json!({
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500
            }),
        );
    }
    if !config.contains_key("modalities") {
        config.insert("modalities".to_string(), json!(["text"]));
    }
    validate_transcription_config(config)
}

fn normalize_translation_config(config: &mut Map<String, Value>) -> Result<(), ApiError> {
    config.insert("type".to_string(), json!("translation"));
    let model = config
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.trim().is_empty())
        .unwrap_or(DEFAULT_TRANSLATION_MODEL)
        .to_string();
    config.insert("model".to_string(), json!(model));
    let audio = config
        .entry("audio".to_string())
        .or_insert_with(|| json!({}));
    let audio = audio
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("translation audio must be an object"))?;
    let input = audio
        .entry("input".to_string())
        .or_insert_with(|| json!({}));
    let input = input
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("translation audio.input must be an object"))?;
    let transcription = input
        .entry("transcription".to_string())
        .or_insert_with(|| json!({}));
    let transcription = transcription.as_object_mut().ok_or_else(|| {
        ApiError::bad_request("translation audio.input.transcription must be an object")
    })?;
    if !transcription.contains_key("model") {
        transcription.insert(
            "model".to_string(),
            json!(DEFAULT_TRANSLATION_TRANSCRIPTION_MODEL),
        );
    }
    input
        .entry("noise_reduction".to_string())
        .or_insert(Value::Null);
    let output = audio
        .entry("output".to_string())
        .or_insert_with(|| json!({}));
    let output = output
        .as_object_mut()
        .ok_or_else(|| ApiError::bad_request("translation audio.output must be an object"))?;
    if !output.contains_key("language") {
        output.insert("language".to_string(), json!(DEFAULT_TRANSLATION_LANGUAGE));
    }
    validate_translation_config(config)
}

fn validate_realtime_config(config: &Map<String, Value>) -> Result<(), ApiError> {
    validate_audio_format(config, "input_audio_format")?;
    validate_audio_format(config, "output_audio_format")?;
    validate_modalities(config.get("modalities"), true)?;
    validate_modalities(config.get("output_modalities"), false)?;
    if let Some(temperature) = config.get("temperature").and_then(Value::as_f64)
        && !(0.6..=1.2).contains(&temperature)
    {
        return Err(ApiError::bad_request(
            "realtime temperature must be between 0.6 and 1.2",
        ));
    }
    if let Some(speed) = config.get("speed").and_then(Value::as_f64)
        && !(0.25..=1.5).contains(&speed)
    {
        return Err(ApiError::bad_request(
            "realtime speed must be between 0.25 and 1.5",
        ));
    }
    if let Some(max_tokens) = config
        .get("max_output_tokens")
        .or_else(|| config.get("max_response_output_tokens"))
    {
        validate_max_tokens(max_tokens)?;
    }
    Ok(())
}

fn validate_transcription_config(config: &Map<String, Value>) -> Result<(), ApiError> {
    validate_audio_format(config, "input_audio_format")?;
    validate_modalities(config.get("modalities"), true)?;
    Ok(())
}

fn validate_translation_config(config: &Map<String, Value>) -> Result<(), ApiError> {
    match config.get("model") {
        Some(Value::String(model)) if !model.trim().is_empty() => {}
        Some(Value::String(_)) => {
            return Err(ApiError::bad_request("translation model must not be empty"));
        }
        Some(_) => return Err(ApiError::bad_request("translation model must be a string")),
        None => {}
    }
    let audio = config
        .get("audio")
        .and_then(Value::as_object)
        .ok_or_else(|| ApiError::bad_request("translation audio must be an object"))?;
    let input = audio
        .get("input")
        .and_then(Value::as_object)
        .ok_or_else(|| ApiError::bad_request("translation audio.input must be an object"))?;
    let transcription = input
        .get("transcription")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ApiError::bad_request("translation audio.input.transcription must be an object")
        })?;
    match transcription.get("model") {
        Some(Value::String(model)) if !model.trim().is_empty() => {}
        Some(Value::String(_)) => {
            return Err(ApiError::bad_request(
                "translation transcription model must not be empty",
            ));
        }
        Some(_) => {
            return Err(ApiError::bad_request(
                "translation transcription model must be a string",
            ));
        }
        None => {}
    }
    if let Some(noise_reduction) = input.get("noise_reduction")
        && !noise_reduction.is_null()
    {
        let ty = noise_reduction
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::bad_request("translation noise_reduction requires type"))?;
        if !matches!(ty, "near_field" | "far_field") {
            return Err(ApiError::bad_request(
                "translation noise_reduction type must be near_field or far_field",
            ));
        }
    }
    let output = audio
        .get("output")
        .and_then(Value::as_object)
        .ok_or_else(|| ApiError::bad_request("translation audio.output must be an object"))?;
    match output.get("language") {
        Some(Value::String(language)) if !language.trim().is_empty() => Ok(()),
        Some(Value::String(_)) => Err(ApiError::bad_request(
            "translation output language must not be empty",
        )),
        Some(_) => Err(ApiError::bad_request(
            "translation output language must be a string",
        )),
        None => Err(ApiError::bad_request(
            "translation audio.output.language is required",
        )),
    }
}

fn validate_audio_format(config: &Map<String, Value>, field: &'static str) -> Result<(), ApiError> {
    match config.get(field) {
        Some(Value::String(value))
            if matches!(value.as_str(), "pcm16" | "g711_ulaw" | "g711_alaw") =>
        {
            Ok(())
        }
        Some(Value::String(_)) => Err(ApiError::bad_request(format!(
            "{field} must be pcm16, g711_ulaw, or g711_alaw"
        ))),
        Some(Value::Null) | None => Ok(()),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be a string"))),
    }
}

fn validate_modalities(value: Option<&Value>, allow_multiple: bool) -> Result<(), ApiError> {
    let Some(value) = value else {
        return Ok(());
    };
    let values = value
        .as_array()
        .ok_or_else(|| ApiError::bad_request("realtime modalities must be an array"))?;
    if !allow_multiple && values.len() > 1 {
        return Err(ApiError::bad_request(
            "realtime output_modalities must contain one modality",
        ));
    }
    for value in values {
        match value.as_str() {
            Some("text" | "audio") => {}
            _ => {
                return Err(ApiError::bad_request(
                    "realtime modalities must contain only text or audio",
                ));
            }
        }
    }
    Ok(())
}

fn validate_max_tokens(value: &Value) -> Result<(), ApiError> {
    match value {
        Value::String(value) if value == "inf" => Ok(()),
        Value::Number(number) => {
            let Some(tokens) = number.as_u64() else {
                return Err(ApiError::bad_request(
                    "realtime max_output_tokens must be an integer or inf",
                ));
            };
            if (1..=4096).contains(&tokens) {
                Ok(())
            } else {
                Err(ApiError::bad_request(
                    "realtime max_output_tokens must be between 1 and 4096",
                ))
            }
        }
        _ => Err(ApiError::bad_request(
            "realtime max_output_tokens must be an integer or inf",
        )),
    }
}

fn request_u64_field(body: &Value, field: &'static str) -> Result<Option<u64>, ApiError> {
    match body.get(field) {
        Some(Value::Number(number)) => number
            .as_u64()
            .map(Some)
            .ok_or_else(|| ApiError::bad_request(format!("{field} must be an integer"))),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be an integer"))),
    }
}

pub(crate) fn translation_secret_expires_at(body: &Value, now: u64) -> Result<u64, ApiError> {
    let seconds = match body.get("expires_after") {
        Some(Value::Object(expires_after)) => {
            match expires_after.get("anchor") {
                Some(Value::String(anchor)) if anchor == "created_at" => {}
                Some(Value::String(_)) => {
                    return Err(ApiError::bad_request(
                        "expires_after.anchor must be created_at",
                    ));
                }
                Some(_) => {
                    return Err(ApiError::bad_request(
                        "expires_after.anchor must be a string",
                    ));
                }
                None => {}
            }
            match expires_after.get("seconds") {
                Some(Value::Number(number)) => number.as_u64().ok_or_else(|| {
                    ApiError::bad_request("expires_after.seconds must be an integer")
                })?,
                Some(_) => {
                    return Err(ApiError::bad_request(
                        "expires_after.seconds must be an integer",
                    ));
                }
                None => TRANSLATION_SECRET_TTL_SECONDS,
            }
        }
        Some(Value::Null) | None => TRANSLATION_SECRET_TTL_SECONDS,
        Some(_) => return Err(ApiError::bad_request("expires_after must be an object")),
    };
    if !(60..=7200).contains(&seconds) {
        return Err(ApiError::bad_request(
            "expires_after.seconds must be between 60 and 7200",
        ));
    }
    Ok(now.saturating_add(seconds))
}

fn optional_nonempty_string(body: &Value, field: &'static str) -> Result<Option<String>, ApiError> {
    match body.get(field) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.clone())),
        Some(Value::String(_)) => Err(ApiError::bad_request(format!("{field} must not be empty"))),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be a string"))),
    }
}

fn upsert_realtime_call(
    state: &AppState,
    call_id: String,
    status: &str,
    session_id: Option<String>,
    status_code: Option<u16>,
    target_uri: Option<String>,
    config: Value,
) -> Result<RealtimeCallRecord, ApiError> {
    let now = unix_seconds();
    let mut calls = lock_realtime_calls(state)?;
    let record = calls
        .entry(call_id.clone())
        .or_insert_with(|| RealtimeCallRecord {
            id: call_id,
            object: "realtime.call",
            created_at: now,
            updated_at: now,
            status: "incoming".to_string(),
            session_id: None,
            status_code: None,
            target_uri: None,
            config: Value::Null,
        });
    record.updated_at = now;
    record.status = status.to_string();
    if session_id.is_some() {
        record.session_id = session_id;
    }
    if status_code.is_some() {
        record.status_code = status_code;
    }
    if target_uri.is_some() {
        record.target_uri = target_uri;
    }
    if !config.is_null() {
        record.config = config;
    }
    Ok(record.clone())
}

fn lock_realtime_sessions(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, RealtimeSessionRecord>>, ApiError> {
    state
        .realtime_sessions
        .lock()
        .map_err(|_| ApiError::internal("realtime session registry lock poisoned"))
}

fn lock_realtime_calls(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, RealtimeCallRecord>>, ApiError> {
    state
        .realtime_calls
        .lock()
        .map_err(|_| ApiError::internal("realtime call registry lock poisoned"))
}

fn realtime_client_secret_value(session_id: &str, created_at: u64) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ created_at;
    for byte in session_id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    format!("ek_nerva_{created_at:x}_{hash:016x}")
}

impl RealtimeSessionKind {
    fn object(self) -> &'static str {
        match self {
            RealtimeSessionKind::Realtime => "realtime.session",
            RealtimeSessionKind::Transcription => "realtime.transcription_session",
            RealtimeSessionKind::Translation => "realtime.translation_session",
        }
    }

    fn kind_name(self) -> &'static str {
        match self {
            RealtimeSessionKind::Realtime => "realtime",
            RealtimeSessionKind::Transcription => "transcription",
            RealtimeSessionKind::Translation => "translation",
        }
    }
}
