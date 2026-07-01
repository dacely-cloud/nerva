use std::collections::HashSet;

use serde_json::{Value, json};

use super::AppState;

pub(crate) const OPENAI_FACADE_MODEL_IDS: &[&str] = &[
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o4-mini",
    "o3",
    "o3-mini",
    "o1",
    "o1-mini",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "gpt-realtime",
    "gpt-realtime-1.5",
    "gpt-realtime-2025-08-28",
    "gpt-realtime-mini",
    "gpt-realtime-mini-2025-10-06",
    "gpt-realtime-mini-2025-12-15",
    "gpt-audio-1.5",
    "gpt-audio-mini",
    "gpt-audio-mini-2025-10-06",
    "gpt-audio-mini-2025-12-15",
    "gpt-4o-realtime-preview",
    "gpt-4o-realtime-preview-2024-10-01",
    "gpt-4o-realtime-preview-2024-12-17",
    "gpt-4o-realtime-preview-2025-06-03",
    "gpt-4o-mini-realtime-preview",
    "gpt-4o-mini-realtime-preview-2024-12-17",
    "gpt-image-1.5",
    "gpt-image-1",
    "gpt-image-1-mini",
    "dall-e-3",
    "dall-e-2",
    "sora-2",
    "sora-2-pro",
    "sora-2-2025-10-06",
    "sora-2-pro-2025-10-06",
    "sora-2-2025-12-08",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-mini-tts",
    "tts-1",
    "tts-1-hd",
    "whisper-1",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
    "omni-moderation-latest",
    "omni-moderation-2024-09-26",
    "text-moderation-latest",
    "text-moderation-stable",
    "computer-use-preview",
    "codex-mini-latest",
];

const OPENAI_FACADE_PREFIXES: &[&str] = &[
    "gpt-",
    "o1-",
    "o3-",
    "o4-",
    "dall-e-",
    "tts-",
    "whisper-",
    "text-embedding-",
    "text-moderation-",
    "omni-moderation-",
    "computer-use-",
    "codex-",
];

pub(crate) fn request_model_id(state: &AppState, body: &Value) -> String {
    body.get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.trim().is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| state.config.model_id.clone())
}

pub(crate) fn model_is_available(state: &AppState, model: &str) -> bool {
    model == state.config.model_id
        || openai_facade_model_available(model)
        || fine_tuned_model_alias_served(state, model)
}

pub(crate) fn openai_facade_model_available(model: &str) -> bool {
    let model = model.trim();
    !model.is_empty()
        && model
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_'))
        && (OPENAI_FACADE_MODEL_IDS.contains(&model)
            || OPENAI_FACADE_PREFIXES
                .iter()
                .any(|prefix| model.starts_with(prefix))
            || matches!(model, "o1" | "o3" | "o4"))
}

pub(crate) fn fine_tuned_model_alias_served(state: &AppState, model: &str) -> bool {
    state.fine_tuning_jobs.lock().ok().is_some_and(|jobs| {
        jobs.values()
            .any(|job| job.fine_tuned_model.as_deref() == Some(model))
    })
}

pub(crate) fn model_record_value(state: &AppState, model: &str) -> Option<Value> {
    let owned_by = if model == state.config.model_id || fine_tuned_model_alias_served(state, model)
    {
        "nerva"
    } else if openai_facade_model_available(model) {
        "openai-facade"
    } else {
        return None;
    };
    Some(model_json(model, owned_by))
}

pub(crate) fn list_model_values(state: &AppState) -> Vec<Value> {
    let mut seen = HashSet::new();
    let mut values = Vec::new();
    push_model_value(&mut values, &mut seen, &state.config.model_id, "nerva");
    if let Ok(jobs) = state.fine_tuning_jobs.lock() {
        for model in jobs
            .values()
            .filter_map(|job| job.fine_tuned_model.as_deref())
        {
            push_model_value(&mut values, &mut seen, model, "nerva");
        }
    }
    for model in OPENAI_FACADE_MODEL_IDS {
        push_model_value(&mut values, &mut seen, model, "openai-facade");
    }
    values
}

pub(crate) fn delete_fine_tuned_model_alias(state: &AppState, model: &str) -> bool {
    let Ok(mut jobs) = state.fine_tuning_jobs.lock() else {
        return false;
    };
    let mut deleted = false;
    for job in jobs.values_mut() {
        if job.fine_tuned_model.as_deref() == Some(model) {
            job.fine_tuned_model = None;
            deleted = true;
        }
    }
    deleted
}

fn push_model_value(values: &mut Vec<Value>, seen: &mut HashSet<String>, model: &str, owner: &str) {
    if seen.insert(model.to_string()) {
        values.push(model_json(model, owner));
    }
}

fn model_json(model: &str, owned_by: &str) -> Value {
    json!({
        "id": model,
        "object": "model",
        "created": 0,
        "owned_by": owned_by
    })
}
