use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{ApiError, AppState, authorize};

const MODERATION_CATEGORIES: [&str; 13] = [
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "illicit",
    "illicit/violent",
    "self-harm",
    "self-harm/instructions",
    "self-harm/intent",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ModerationInput {
    pub(crate) text: String,
    pub(crate) input_types: Vec<&'static str>,
}

pub(crate) async fn moderations(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(create_moderation_response(&state, &body)?))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn create_moderation_response(
    state: &AppState,
    body: &Value,
) -> Result<Value, ApiError> {
    let model = request_moderation_model(body)?;
    let inputs = parse_moderation_inputs(body)?;
    let results = inputs
        .iter()
        .map(|input| moderation_result_for_text(&input.text, &input.input_types))
        .collect::<Vec<_>>();
    Ok(json!({
        "id": state.next_response_id("modr"),
        "model": model,
        "results": results
    }))
}

pub(crate) fn parse_moderation_inputs(body: &Value) -> Result<Vec<ModerationInput>, ApiError> {
    let input = body
        .get("input")
        .ok_or_else(|| ApiError::bad_request("moderation request requires input"))?;
    match input {
        Value::String(text) => Ok(vec![ModerationInput::text(text.clone())]),
        Value::Array(items) => {
            if items.is_empty() {
                return Err(ApiError::bad_request("moderation input must not be empty"));
            }
            if items.iter().all(Value::is_string) {
                return Ok(items
                    .iter()
                    .map(|item| ModerationInput::text(item.as_str().unwrap_or("").to_string()))
                    .collect());
            }
            Ok(vec![moderation_input_from_value(input)?])
        }
        Value::Object(_) => Ok(vec![moderation_input_from_value(input)?]),
        Value::Null => Err(ApiError::bad_request("moderation input must not be null")),
        _ => Err(ApiError::bad_request(
            "moderation input must be a string, string array, or content part array",
        )),
    }
}

pub(crate) fn moderation_result_for_text(text: &str, input_types: &[&'static str]) -> Value {
    let text = text.to_ascii_lowercase();
    let mut scores = BTreeMap::new();
    for category in MODERATION_CATEGORIES {
        scores.insert(category, 0.0f64);
    }
    score_categories(&text, &mut scores);
    let categories = scores
        .iter()
        .map(|(category, score)| ((*category).to_string(), json!(*score >= 0.5)))
        .collect::<serde_json::Map<_, _>>();
    let category_scores = scores
        .iter()
        .map(|(category, score)| ((*category).to_string(), json!(score)))
        .collect::<serde_json::Map<_, _>>();
    let applied_input_types = MODERATION_CATEGORIES
        .into_iter()
        .map(|category| (category.to_string(), json!(input_types)))
        .collect::<serde_json::Map<_, _>>();
    let flagged = scores.values().any(|score| *score >= 0.5);
    json!({
        "flagged": flagged,
        "categories": categories,
        "category_scores": category_scores,
        "category_applied_input_types": applied_input_types
    })
}

impl ModerationInput {
    fn text(text: String) -> Self {
        Self {
            text,
            input_types: vec!["text"],
        }
    }
}

fn request_moderation_model(body: &Value) -> Result<String, ApiError> {
    match body.get("model") {
        Some(Value::String(model)) if !model.trim().is_empty() => Ok(model.clone()),
        Some(Value::String(_)) => Err(ApiError::bad_request("moderation model must not be empty")),
        Some(Value::Null) | None => Ok("omni-moderation-latest".to_string()),
        Some(_) => Err(ApiError::bad_request("moderation model must be a string")),
    }
}

fn moderation_input_from_value(value: &Value) -> Result<ModerationInput, ApiError> {
    let mut text = String::new();
    let mut has_text = false;
    let mut has_image = false;
    collect_moderation_input(value, &mut text, &mut has_text, &mut has_image)?;
    let mut input_types = Vec::new();
    if has_text {
        input_types.push("text");
    }
    if has_image {
        input_types.push("image");
    }
    if input_types.is_empty() {
        return Err(ApiError::bad_request(
            "moderation content parts must contain text or image input",
        ));
    }
    Ok(ModerationInput { text, input_types })
}

fn collect_moderation_input(
    value: &Value,
    text: &mut String,
    has_text: &mut bool,
    has_image: &mut bool,
) -> Result<(), ApiError> {
    match value {
        Value::String(value) => {
            append_text(text, value);
            *has_text = true;
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                collect_moderation_input(item, text, has_text, has_image)?;
            }
            Ok(())
        }
        Value::Object(object) => {
            let input_type = object.get("type").and_then(Value::as_str).unwrap_or("");
            if matches!(input_type, "image" | "image_url" | "input_image")
                || object.get("image_url").is_some()
                || object.get("image").is_some()
            {
                *has_image = true;
            }
            if let Some(value) = object
                .get("text")
                .or_else(|| object.get("input_text"))
                .or_else(|| object.get("content"))
                .or_else(|| object.get("input"))
            {
                collect_moderation_input(value, text, has_text, has_image)?;
            } else if matches!(input_type, "text" | "input_text") {
                *has_text = true;
            }
            Ok(())
        }
        Value::Null => Ok(()),
        _ => Err(ApiError::bad_request(
            "moderation content parts must be strings or objects",
        )),
    }
}

fn append_text(out: &mut String, text: &str) {
    if !out.is_empty() && !text.is_empty() {
        out.push('\n');
    }
    out.push_str(text);
}

fn score_categories(text: &str, scores: &mut BTreeMap<&'static str, f64>) {
    if contains_any(
        text,
        &[
            "harass",
            "bully",
            "threaten you",
            "you are worthless",
            "go away forever",
        ],
    ) {
        set_score(scores, "harassment", 0.85);
    }
    if contains_any(
        text,
        &["i will kill you", "i will hurt you", "hurt you badly"],
    ) {
        set_score(scores, "harassment", 0.9);
        set_score(scores, "harassment/threatening", 0.92);
        set_score(scores, "violence", 0.8);
    }
    if contains_any(
        text,
        &[
            "hate group",
            "racial slur",
            "ethnic slur",
            "dehumanize",
            "inferior race",
        ],
    ) {
        set_score(scores, "hate", 0.86);
    }
    if contains_any(text, &["kill all", "exterminate", "violent hate"]) {
        set_score(scores, "hate", 0.9);
        set_score(scores, "hate/threatening", 0.92);
        set_score(scores, "violence", 0.85);
    }
    if contains_any(
        text,
        &[
            "shoplift",
            "steal",
            "fraud",
            "phishing",
            "malware",
            "cocaine",
            "illegal drugs",
            "bypass security",
        ],
    ) {
        set_score(scores, "illicit", 0.84);
    }
    if contains_any(
        text,
        &[
            "build a bomb",
            "make a bomb",
            "buy a gun illegally",
            "procure a weapon",
        ],
    ) {
        set_score(scores, "illicit", 0.9);
        set_score(scores, "illicit/violent", 0.93);
        set_score(scores, "violence", 0.86);
    }
    if contains_any(
        text,
        &[
            "suicide",
            "self harm",
            "self-harm",
            "hurt myself",
            "kill myself",
            "end my life",
        ],
    ) {
        set_score(scores, "self-harm", 0.9);
    }
    if contains_any(
        text,
        &[
            "how to kill myself",
            "how to self harm",
            "instructions for suicide",
        ],
    ) {
        set_score(scores, "self-harm", 0.95);
        set_score(scores, "self-harm/instructions", 0.96);
    }
    if contains_any(
        text,
        &[
            "i want to die",
            "i am going to kill myself",
            "i will end my life",
        ],
    ) {
        set_score(scores, "self-harm", 0.95);
        set_score(scores, "self-harm/intent", 0.96);
    }
    if contains_any(
        text,
        &[
            "sexual content",
            "porn",
            "explicit sex",
            "nude photo",
            "sexual services",
        ],
    ) {
        set_score(scores, "sexual", 0.86);
    }
    if contains_any(text, &["underage sexual", "sexual minor", "child sexual"]) {
        set_score(scores, "sexual", 0.95);
        set_score(scores, "sexual/minors", 0.98);
    }
    if contains_any(
        text,
        &[
            "kill", "murder", "assault", "shoot", "stab", "attack", "weapon", "bomb",
        ],
    ) {
        set_score(scores, "violence", 0.82);
    }
    if contains_any(
        text,
        &["gore", "dismember", "graphic injury", "blood everywhere"],
    ) {
        set_score(scores, "violence", 0.9);
        set_score(scores, "violence/graphic", 0.92);
    }
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn set_score(scores: &mut BTreeMap<&'static str, f64>, category: &'static str, score: f64) {
    let current = scores.entry(category).or_insert(0.0);
    if score > *current {
        *current = score;
    }
}
