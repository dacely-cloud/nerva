use std::sync::atomic::Ordering;

use actix_web::{HttpRequest, HttpResponse, web};
use nerva_model::hf::tokenizer::{PromptFormat, encode_text_prompt, format_prompt_for_model};
use serde_json::{Value, json};

use super::{ApiError, AppState, authorize};

const DEFAULT_EMBEDDING_DIMENSIONS: usize = 1536;
const MAX_EMBEDDING_DIMENSIONS: usize = 8192;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EmbeddingInput {
    Text(String),
    TokenIds(Vec<u32>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EmbeddingEncodingFormat {
    Float,
    Base64,
}

pub(crate) async fn embeddings(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        let response = create_embeddings_response(&state, &body)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(response))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn create_embeddings_response(
    state: &AppState,
    body: &Value,
) -> Result<Value, ApiError> {
    let model = request_embedding_model(state, body)?;
    let inputs = parse_embedding_inputs(body)?;
    let dimensions = request_embedding_dimensions(body, &model)?;
    let encoding_format = request_embedding_encoding_format(body)?;
    let mut prompt_tokens = 0usize;
    let mut data = Vec::with_capacity(inputs.len());
    for (index, input) in inputs.iter().enumerate() {
        prompt_tokens = prompt_tokens.saturating_add(embedding_token_count(state, input)?);
        let embedding = deterministic_embedding_vector(&model, input, dimensions);
        data.push(json!({
            "object": "embedding",
            "embedding": embedding_json(&embedding, encoding_format),
            "index": index
        }));
    }
    Ok(json!({
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens
        }
    }))
}

pub(crate) fn parse_embedding_inputs(body: &Value) -> Result<Vec<EmbeddingInput>, ApiError> {
    let input = body
        .get("input")
        .ok_or_else(|| ApiError::bad_request("embeddings request requires input"))?;
    match input {
        Value::String(text) => Ok(vec![EmbeddingInput::Text(text.clone())]),
        Value::Array(items) => parse_embedding_array_input(items),
        Value::Null => Err(ApiError::bad_request("embedding input must not be null")),
        _ => Err(ApiError::bad_request(
            "embedding input must be a string, string array, token array, or token-array array",
        )),
    }
}

pub(crate) fn deterministic_embedding_vector(
    model: &str,
    input: &EmbeddingInput,
    dimensions: usize,
) -> Vec<f32> {
    let mut vector = Vec::with_capacity(dimensions);
    let fingerprint = embedding_fingerprint(model, input);
    for index in 0..dimensions {
        let hash = hash_embedding_slot(&fingerprint, index as u64);
        let mantissa = ((hash >> 40) as u32) as f32 / ((1u32 << 24) as f32);
        vector.push(mantissa.mul_add(2.0, -1.0));
    }
    normalize_vector(&mut vector);
    vector
}

fn parse_embedding_array_input(items: &[Value]) -> Result<Vec<EmbeddingInput>, ApiError> {
    if items.is_empty() {
        return Err(ApiError::bad_request(
            "embedding input array must not be empty",
        ));
    }
    if items.iter().all(Value::is_string) {
        return Ok(items
            .iter()
            .map(|item| EmbeddingInput::Text(item.as_str().unwrap_or("").to_string()))
            .collect());
    }
    if items.iter().all(Value::is_u64) {
        return Ok(vec![EmbeddingInput::TokenIds(parse_token_ids(items)?)]);
    }
    if items.iter().all(Value::is_array) {
        return items
            .iter()
            .map(|item| {
                let values = item
                    .as_array()
                    .expect("embedding array input was prevalidated");
                parse_token_ids(values).map(EmbeddingInput::TokenIds)
            })
            .collect();
    }
    Err(ApiError::bad_request(
        "embedding input array must contain only strings, only token ids, or only token-id arrays",
    ))
}

fn parse_token_ids(values: &[Value]) -> Result<Vec<u32>, ApiError> {
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|token| u32::try_from(token).ok())
                .ok_or_else(|| ApiError::bad_request("embedding token ids must be u32 integers"))
        })
        .collect()
}

fn request_embedding_model(state: &AppState, body: &Value) -> Result<String, ApiError> {
    match body.get("model") {
        Some(Value::String(model)) if !model.trim().is_empty() => Ok(model.clone()),
        Some(Value::String(_)) => Err(ApiError::bad_request("embedding model must not be empty")),
        Some(Value::Null) | None => Ok(state.config.model_id.clone()),
        Some(_) => Err(ApiError::bad_request("embedding model must be a string")),
    }
}

fn request_embedding_dimensions(body: &Value, model: &str) -> Result<usize, ApiError> {
    match body.get("dimensions") {
        Some(Value::Number(number)) => {
            let dimensions = number
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| ApiError::bad_request("embedding dimensions must be an integer"))?;
            if dimensions == 0 || dimensions > MAX_EMBEDDING_DIMENSIONS {
                Err(ApiError::bad_request(format!(
                    "embedding dimensions must be between 1 and {MAX_EMBEDDING_DIMENSIONS}"
                )))
            } else {
                Ok(dimensions)
            }
        }
        Some(Value::Null) | None => Ok(default_embedding_dimensions(model)),
        Some(_) => Err(ApiError::bad_request(
            "embedding dimensions must be an integer",
        )),
    }
}

fn request_embedding_encoding_format(body: &Value) -> Result<EmbeddingEncodingFormat, ApiError> {
    match body.get("encoding_format") {
        Some(Value::String(value)) if value == "float" => Ok(EmbeddingEncodingFormat::Float),
        Some(Value::String(value)) if value == "base64" => Ok(EmbeddingEncodingFormat::Base64),
        Some(Value::String(_)) => Err(ApiError::bad_request(
            "embedding encoding_format must be float or base64",
        )),
        Some(Value::Null) | None => Ok(EmbeddingEncodingFormat::Float),
        Some(_) => Err(ApiError::bad_request(
            "embedding encoding_format must be a string",
        )),
    }
}

fn default_embedding_dimensions(model: &str) -> usize {
    match model {
        "text-embedding-3-large" => 3072,
        "text-embedding-3-small" | "text-embedding-ada-002" => 1536,
        _ => DEFAULT_EMBEDDING_DIMENSIONS,
    }
}

fn embedding_token_count(state: &AppState, input: &EmbeddingInput) -> Result<usize, ApiError> {
    match input {
        EmbeddingInput::Text(text) => {
            let formatted =
                format_prompt_for_model(&state.config.model_path, text, PromptFormat::Raw)
                    .map_err(ApiError::bad_request)?;
            let encoded = encode_text_prompt(&state.config.model_path, &formatted.text)
                .map_err(ApiError::bad_request)?;
            Ok(encoded.token_ids.len())
        }
        EmbeddingInput::TokenIds(tokens) => Ok(tokens.len()),
    }
}

fn embedding_json(embedding: &[f32], encoding_format: EmbeddingEncodingFormat) -> Value {
    match encoding_format {
        EmbeddingEncodingFormat::Float => json!(embedding),
        EmbeddingEncodingFormat::Base64 => json!(embedding_base64(embedding)),
    }
}

fn embedding_base64(embedding: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(embedding.len().saturating_mul(4));
    for value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    base64_encode(&bytes)
}

fn embedding_fingerprint(model: &str, input: &EmbeddingInput) -> Vec<u8> {
    let mut fingerprint = Vec::new();
    fingerprint.extend_from_slice(model.as_bytes());
    fingerprint.push(0);
    match input {
        EmbeddingInput::Text(text) => {
            fingerprint.push(b't');
            fingerprint.extend_from_slice(text.as_bytes());
        }
        EmbeddingInput::TokenIds(tokens) => {
            fingerprint.push(b'i');
            for token in tokens {
                fingerprint.extend_from_slice(&token.to_le_bytes());
            }
        }
    }
    fingerprint
}

fn hash_embedding_slot(bytes: &[u8], slot: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ slot.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
        hash ^= hash >> 32;
    }
    hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^ (hash >> 31)
}

fn normalize_vector(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if norm <= f64::EPSILON {
        return;
    }
    for value in vector {
        *value = (f64::from(*value) / norm) as f32;
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3).saturating_mul(4));
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        let n = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((n >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(n & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}
