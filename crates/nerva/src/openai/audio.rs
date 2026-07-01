use std::sync::atomic::Ordering;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{ApiError, AppState, authorize};

const DEFAULT_TRANSCRIPTION_MODEL: &str = "gpt-4o-transcribe";
const DEFAULT_TTS_MODEL: &str = "gpt-4o-mini-tts";
const DEFAULT_TTS_VOICE: &str = "alloy";
const DEFAULT_SAMPLE_RATE: u32 = 16_000;

#[derive(Clone, Debug)]
pub(crate) struct ParsedAudioRequest {
    pub(crate) model: String,
    pub(crate) filename: String,
    pub(crate) content: Vec<u8>,
    pub(crate) prompt: Option<String>,
    pub(crate) response_format: String,
    pub(crate) language: Option<String>,
    pub(crate) temperature: Option<f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct ParsedSpeechRequest {
    pub(crate) model: String,
    pub(crate) input: String,
    pub(crate) voice: String,
    pub(crate) response_format: String,
    pub(crate) speed: f32,
}

pub(crate) async fn create_audio_transcription(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        let parsed = parse_audio_request(&request, &body)?;
        let response = audio_text_response(&parsed, false)?;
        Ok::<_, ApiError>(response)
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_audio_translation(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        let mut parsed = parse_audio_request(&request, &body)?;
        parsed.language = Some("english".to_string());
        let response = audio_text_response(&parsed, true)?;
        Ok::<_, ApiError>(response)
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn create_audio_speech(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        let parsed = parse_speech_request(&body)?;
        let (content_type, speech_body) = speech_response_bytes(&parsed)?;
        Ok::<_, ApiError>(
            HttpResponse::Ok()
                .insert_header(("x-nerva-audio-model", parsed.model))
                .insert_header(("x-nerva-audio-voice", parsed.voice))
                .insert_header(("x-nerva-audio-format", parsed.response_format))
                .content_type(content_type)
                .body(speech_body),
        )
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn parse_audio_request(
    request: &HttpRequest,
    body: &[u8],
) -> Result<ParsedAudioRequest, ApiError> {
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    let fields = if content_type.starts_with("multipart/form-data") {
        let boundary = multipart_boundary(content_type)
            .ok_or_else(|| ApiError::bad_request("multipart audio request is missing boundary"))?;
        parse_multipart_audio_fields(body, &boundary)?
    } else if content_type.starts_with("application/json") || looks_like_json(body) {
        let value: Value = serde_json::from_slice(body)
            .map_err(|err| ApiError::bad_request(format!("invalid audio JSON request: {err}")))?;
        return parse_audio_json_request(&value);
    } else {
        ParsedAudioFields {
            filename: query_param(request.query_string(), "filename")
                .unwrap_or_else(|| "audio.bin".to_string()),
            content: body.to_vec(),
            model: query_param(request.query_string(), "model"),
            prompt: query_param(request.query_string(), "prompt"),
            response_format: query_param(request.query_string(), "response_format"),
            language: query_param(request.query_string(), "language"),
            temperature: query_param(request.query_string(), "temperature")
                .and_then(|value| value.parse::<f32>().ok()),
        }
    };
    normalize_audio_fields(fields)
}

pub(crate) fn parse_audio_json_request(body: &Value) -> Result<ParsedAudioRequest, ApiError> {
    let content = body
        .get("file")
        .or_else(|| body.get("audio"))
        .or_else(|| body.get("content"))
        .or_else(|| body.get("data"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ApiError::bad_request("audio JSON request requires file, audio, content, or data")
        })?
        .as_bytes()
        .to_vec();
    let fields = ParsedAudioFields {
        filename: body
            .get("filename")
            .and_then(Value::as_str)
            .unwrap_or("audio.json")
            .to_string(),
        content,
        model: body
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
        prompt: body
            .get("prompt")
            .and_then(Value::as_str)
            .map(str::to_string),
        response_format: body
            .get("response_format")
            .and_then(Value::as_str)
            .map(str::to_string),
        language: body
            .get("language")
            .and_then(Value::as_str)
            .map(str::to_string),
        temperature: body
            .get("temperature")
            .and_then(Value::as_f64)
            .map(|value| value as f32),
    };
    normalize_audio_fields(fields)
}

fn normalize_audio_fields(fields: ParsedAudioFields) -> Result<ParsedAudioRequest, ApiError> {
    if fields.content.is_empty() {
        return Err(ApiError::bad_request(
            "audio request requires non-empty file",
        ));
    }
    Ok(ParsedAudioRequest {
        model: fields
            .model
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_TRANSCRIPTION_MODEL.to_string()),
        filename: fields.filename,
        content: fields.content,
        prompt: fields.prompt.filter(|value| !value.trim().is_empty()),
        response_format: fields
            .response_format
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| "json".to_string()),
        language: fields.language.filter(|value| !value.trim().is_empty()),
        temperature: fields.temperature,
    })
}

pub(crate) fn parse_speech_request(body: &Value) -> Result<ParsedSpeechRequest, ApiError> {
    let input = required_nonempty_string(body, "input")?;
    let model =
        optional_nonempty_string(body, "model")?.unwrap_or_else(|| DEFAULT_TTS_MODEL.into());
    let voice =
        optional_nonempty_string(body, "voice")?.unwrap_or_else(|| DEFAULT_TTS_VOICE.into());
    let response_format =
        optional_nonempty_string(body, "response_format")?.unwrap_or_else(|| "mp3".into());
    let speed = match body.get("speed") {
        Some(Value::Number(number)) => number
            .as_f64()
            .filter(|speed| *speed > 0.0 && *speed <= 4.0)
            .ok_or_else(|| ApiError::bad_request("speech speed must be between 0 and 4"))?
            as f32,
        Some(Value::Null) | None => 1.0,
        Some(_) => return Err(ApiError::bad_request("speech speed must be a number")),
    };
    Ok(ParsedSpeechRequest {
        model,
        input,
        voice,
        response_format,
        speed,
    })
}

pub(crate) fn audio_text_response_value(
    parsed: &ParsedAudioRequest,
    translated: bool,
) -> Result<Value, ApiError> {
    let text = inferred_audio_text(parsed, translated);
    match parsed.response_format.as_str() {
        "json" => Ok(json!({
            "text": text,
            "usage": audio_usage(parsed)
        })),
        "verbose_json" | "diarized_json" => Ok(verbose_audio_json(parsed, &text, translated)),
        "text" => Ok(Value::String(text)),
        "srt" => Ok(Value::String(format!(
            "1\n00:00:00,000 --> 00:00:01,000\n{text}\n"
        ))),
        "vtt" => Ok(Value::String(format!(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n{text}\n"
        ))),
        other => Err(ApiError::bad_request(format!(
            "unsupported audio response_format '{other}'"
        ))),
    }
}

pub(crate) fn create_audio_text_batch_response(
    body: &Value,
    translated: bool,
) -> Result<Value, ApiError> {
    let mut parsed = parse_audio_json_request(body)?;
    if translated {
        parsed.language = Some("english".to_string());
    }
    audio_text_response_value(&parsed, translated)
}

pub(crate) fn create_speech_batch_response(body: &Value) -> Result<Value, ApiError> {
    let parsed = parse_speech_request(body)?;
    let (content_type, speech_body) = speech_response_bytes(&parsed)?;
    Ok(json!({
        "model": parsed.model,
        "voice": parsed.voice,
        "response_format": parsed.response_format,
        "content_type": content_type,
        "encoding": "base64",
        "data": base64_bytes(&speech_body)
    }))
}

#[derive(Clone, Debug)]
struct ParsedAudioFields {
    filename: String,
    content: Vec<u8>,
    model: Option<String>,
    prompt: Option<String>,
    response_format: Option<String>,
    language: Option<String>,
    temperature: Option<f32>,
}

fn audio_text_response(
    parsed: &ParsedAudioRequest,
    translated: bool,
) -> Result<HttpResponse, ApiError> {
    let value = audio_text_response_value(parsed, translated)?;
    match parsed.response_format.as_str() {
        "json" => Ok(HttpResponse::Ok()
            .insert_header(("x-nerva-audio-model", parsed.model.clone()))
            .json(value)),
        "verbose_json" | "diarized_json" => Ok(HttpResponse::Ok()
            .insert_header(("x-nerva-audio-model", parsed.model.clone()))
            .json(value)),
        "text" => Ok(HttpResponse::Ok()
            .insert_header(("x-nerva-audio-model", parsed.model.clone()))
            .content_type("text/plain; charset=utf-8")
            .body(value.as_str().unwrap_or_default().to_string())),
        "srt" => Ok(HttpResponse::Ok()
            .insert_header(("x-nerva-audio-model", parsed.model.clone()))
            .content_type("application/x-subrip; charset=utf-8")
            .body(value.as_str().unwrap_or_default().to_string())),
        "vtt" => Ok(HttpResponse::Ok()
            .insert_header(("x-nerva-audio-model", parsed.model.clone()))
            .content_type("text/vtt; charset=utf-8")
            .body(value.as_str().unwrap_or_default().to_string())),
        other => Err(ApiError::bad_request(format!(
            "unsupported audio response_format '{other}'"
        ))),
    }
}

fn inferred_audio_text(parsed: &ParsedAudioRequest, translated: bool) -> String {
    if let Some(prompt) = parsed.prompt.as_deref() {
        return prompt.to_string();
    }
    let lossy = String::from_utf8_lossy(&parsed.content);
    let printable = lossy
        .chars()
        .filter(|ch| ch.is_ascii_graphic() || ch.is_ascii_whitespace())
        .collect::<String>();
    let printable = printable.trim();
    if !printable.is_empty()
        && printable.len() >= parsed.content.len().saturating_div(2).clamp(1, 4096)
    {
        return printable.chars().take(4096).collect();
    }
    if translated {
        format!("[English translation unavailable for {}]", parsed.filename)
    } else {
        format!("[Audio transcription unavailable for {}]", parsed.filename)
    }
}

fn verbose_audio_json(parsed: &ParsedAudioRequest, text: &str, translated: bool) -> Value {
    let duration = estimated_audio_duration_seconds(parsed.content.len());
    let segment = json!({
        "id": 0,
        "seek": 0,
        "start": 0.0,
        "end": duration,
        "text": text,
        "tokens": [],
        "temperature": parsed.temperature.unwrap_or(0.0),
        "avg_logprob": 0.0,
        "compression_ratio": 0.0,
        "no_speech_prob": if text.starts_with("[Audio") { 1.0 } else { 0.0 }
    });
    let mut response = json!({
        "task": if translated { "translate" } else { "transcribe" },
        "language": parsed.language.as_deref().unwrap_or(if translated { "english" } else { "unknown" }),
        "duration": duration,
        "text": text,
        "segments": [segment],
        "usage": audio_usage(parsed)
    });
    if parsed.response_format == "diarized_json" {
        response["segments"][0]["speaker"] = json!("speaker_0");
    }
    response
}

fn audio_usage(parsed: &ParsedAudioRequest) -> Value {
    json!({
        "type": "duration",
        "seconds": estimated_audio_duration_seconds(parsed.content.len())
    })
}

pub(crate) fn wav_bytes(pcm: &[i16], sample_rate: u32) -> Vec<u8> {
    let data_bytes = pcm.len().saturating_mul(2);
    let data_bytes_u32 = u32::try_from(data_bytes).unwrap_or(u32::MAX);
    let riff_size = 36u32.saturating_add(data_bytes_u32);
    let mut out = Vec::with_capacity(44usize.saturating_add(data_bytes));
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&sample_rate.saturating_mul(2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes_u32.to_le_bytes());
    for sample in pcm {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

pub(crate) fn pcm_bytes(pcm: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len().saturating_mul(2));
    for sample in pcm {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

pub(crate) fn speech_pcm(input: &str, speed: f32, sample_rate: u32) -> Vec<i16> {
    let chars = input.chars().count().max(1);
    let seconds = ((chars as f32 / 18.0) / speed.max(0.1)).clamp(0.25, 8.0);
    let samples = (seconds * sample_rate as f32) as usize;
    let mut pcm = Vec::with_capacity(samples);
    let seed = input.bytes().fold(0x811c_9dc5u32, |hash, byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    });
    let base_hz = 180.0 + (seed % 220) as f32;
    for index in 0..samples {
        let t = index as f32 / sample_rate as f32;
        let envelope = if index < sample_rate as usize / 100 {
            index as f32 / (sample_rate as f32 / 100.0)
        } else if index + sample_rate as usize / 100 > samples {
            (samples.saturating_sub(index) as f32 / (sample_rate as f32 / 100.0)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let wave = (t * base_hz * std::f32::consts::TAU).sin()
            + 0.35 * (t * base_hz * 2.0 * std::f32::consts::TAU).sin();
        pcm.push((wave * envelope * 8000.0) as i16);
    }
    pcm
}

fn speech_response_bytes(
    parsed: &ParsedSpeechRequest,
) -> Result<(&'static str, Vec<u8>), ApiError> {
    let sample_rate = DEFAULT_SAMPLE_RATE;
    let pcm = speech_pcm(&parsed.input, parsed.speed, sample_rate);
    match parsed.response_format.as_str() {
        "pcm" => Ok(("audio/pcm", pcm_bytes(&pcm))),
        "wav" | "mp3" | "opus" | "aac" | "flac" => Ok(("audio/wav", wav_bytes(&pcm, sample_rate))),
        other => Err(ApiError::bad_request(format!(
            "unsupported speech response_format '{other}'"
        ))),
    }
}

fn base64_bytes(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0b0000_0011) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(((b1 & 0b0000_1111) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(b2 & 0b0011_1111) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn parse_multipart_audio_fields(
    body: &[u8],
    boundary: &str,
) -> Result<ParsedAudioFields, ApiError> {
    let text = String::from_utf8_lossy(body);
    let marker = format!("--{boundary}");
    let mut fields = ParsedAudioFields {
        filename: "audio.bin".to_string(),
        content: Vec::new(),
        model: None,
        prompt: None,
        response_format: None,
        language: None,
        temperature: None,
    };
    for raw_part in text.split(&marker) {
        let part = raw_part.trim_start_matches("\r\n");
        if part.is_empty() || part.starts_with("--") {
            continue;
        }
        let Some((headers, value)) = part.split_once("\r\n\r\n") else {
            continue;
        };
        let disposition = headers
            .lines()
            .find(|line| {
                line.to_ascii_lowercase()
                    .starts_with("content-disposition:")
            })
            .unwrap_or("");
        let Some(name) = disposition_param(disposition, "name") else {
            continue;
        };
        let value = value.strip_suffix("--").unwrap_or(value);
        let value = value.strip_suffix("\r\n").unwrap_or(value);
        match name.as_str() {
            "file" => {
                fields.filename = disposition_param(disposition, "filename")
                    .unwrap_or_else(|| "audio.bin".to_string());
                fields.content = value.as_bytes().to_vec();
            }
            "model" => fields.model = Some(value.trim().to_string()),
            "prompt" => fields.prompt = Some(value.trim().to_string()),
            "response_format" => fields.response_format = Some(value.trim().to_string()),
            "language" => fields.language = Some(value.trim().to_string()),
            "temperature" => fields.temperature = value.trim().parse::<f32>().ok(),
            _ => {}
        }
    }
    Ok(fields)
}

fn required_nonempty_string(body: &Value, field: &'static str) -> Result<String, ApiError> {
    match body.get(field) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(value.clone()),
        Some(Value::String(_)) => Err(ApiError::bad_request(format!("{field} must not be empty"))),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be a string"))),
        None => Err(ApiError::bad_request(format!("{field} is required"))),
    }
}

fn optional_nonempty_string(body: &Value, field: &'static str) -> Result<Option<String>, ApiError> {
    match body.get(field) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.clone())),
        Some(Value::String(_)) => Err(ApiError::bad_request(format!("{field} must not be empty"))),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be a string"))),
    }
}

fn estimated_audio_duration_seconds(bytes: usize) -> f64 {
    ((bytes as f64 / 32_000.0).max(0.001) * 1000.0).round() / 1000.0
}

fn looks_like_json(body: &[u8]) -> bool {
    body.iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace())
        .is_some_and(|byte| byte == b'{' || byte == b'[')
}

fn multipart_boundary(content_type: &str) -> Option<String> {
    content_type
        .split(';')
        .map(str::trim)
        .find_map(|part| part.strip_prefix("boundary="))
        .map(|boundary| boundary.trim_matches('"').to_string())
        .filter(|boundary| !boundary.is_empty())
}

fn disposition_param(disposition: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    disposition
        .split(';')
        .map(str::trim)
        .find_map(|part| part.strip_prefix(&prefix))
        .map(|value| value.trim_matches('"').to_string())
        .filter(|value| !value.is_empty())
}

fn query_param(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|part| {
        let (key, value) = part.split_once('=')?;
        (percent_decode_query(key) == name).then(|| percent_decode_query(value))
    })
}

fn percent_decode_query(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                out.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let hi = hex_value(bytes[index + 1]);
                let lo = hex_value(bytes[index + 2]);
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    out.push((hi << 4) | lo);
                    index += 3;
                } else {
                    out.push(bytes[index]);
                    index += 1;
                }
            }
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).to_string()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
