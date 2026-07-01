use std::sync::atomic::Ordering;

use actix_web::{HttpRequest, HttpResponse, web};
use futures_util::stream;
use serde_json::{Value, json};

use super::{ApiError, AppState, authorize, unix_seconds};

const DEFAULT_IMAGE_MODEL: &str = "gpt-image-1";
const DEFAULT_IMAGE_SIZE: &str = "1024x1024";
const DEFAULT_IMAGE_QUALITY: &str = "auto";
const DEFAULT_IMAGE_OUTPUT_FORMAT: &str = "png";
const MAX_IMAGES_PER_REQUEST: usize = 10;
const MAX_PARTIAL_IMAGES: usize = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ImageTask {
    Generation,
    Edit,
    Variation,
}

#[derive(Clone, Debug)]
pub(crate) struct ParsedImageInput {
    pub(crate) filename: String,
    pub(crate) content: Vec<u8>,
}

#[derive(Clone, Debug)]
pub(crate) struct ParsedImageRequest {
    pub(crate) task: ImageTask,
    pub(crate) model: String,
    pub(crate) prompt: Option<String>,
    pub(crate) images: Vec<ParsedImageInput>,
    pub(crate) mask: Option<ParsedImageInput>,
    pub(crate) n: usize,
    pub(crate) size: String,
    pub(crate) quality: String,
    pub(crate) background: String,
    pub(crate) output_format: String,
    pub(crate) response_format: ImageResponseFormat,
    pub(crate) stream: bool,
    pub(crate) partial_images: usize,
    pub(crate) style: Option<String>,
    pub(crate) moderation: Option<String>,
    pub(crate) output_compression: Option<u8>,
    pub(crate) user: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ImageResponseFormat {
    B64Json,
    Url,
}

pub(crate) async fn create_image_generation(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    create_image(state, request, body, ImageTask::Generation).await
}

pub(crate) async fn create_image_edit(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    create_image(state, request, body, ImageTask::Edit).await
}

pub(crate) async fn create_image_variation(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
) -> HttpResponse {
    create_image(state, request, body, ImageTask::Variation).await
}

async fn create_image(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Bytes,
    task: ImageTask,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        let parsed = parse_image_request(task, &request, &body)?;
        let created = unix_seconds();
        if parsed.stream {
            Ok::<_, ApiError>(image_stream_response(&parsed, created))
        } else {
            Ok::<_, ApiError>(
                HttpResponse::Ok()
                    .insert_header(("x-nerva-image-model", parsed.model.clone()))
                    .insert_header(("x-nerva-image-backend", "deterministic-placeholder"))
                    .json(image_response_value(&parsed, created)),
            )
        }
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

fn parse_image_request(
    task: ImageTask,
    request: &HttpRequest,
    body: &[u8],
) -> Result<ParsedImageRequest, ApiError> {
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if content_type.starts_with("multipart/form-data") {
        let boundary = multipart_boundary(content_type)
            .ok_or_else(|| ApiError::bad_request("multipart image request is missing boundary"))?;
        return parse_multipart_image_request(task, body, &boundary);
    }
    if content_type.starts_with("application/json") || looks_like_json(body) {
        let value: Value = serde_json::from_slice(body)
            .map_err(|err| ApiError::bad_request(format!("invalid image JSON request: {err}")))?;
        return parse_image_json_request(task, &value);
    }
    parse_raw_image_request(task, request, body)
}

pub(crate) fn parse_image_json_request(
    task: ImageTask,
    body: &Value,
) -> Result<ParsedImageRequest, ApiError> {
    let fields = ImageFields {
        model: optional_json_string(body, "model")?,
        prompt: optional_json_string(body, "prompt")?,
        images: json_image_inputs(body.get("image"))?,
        mask: json_image_input(body.get("mask"), "mask", 0)?,
        n: optional_json_usize(body, "n")?,
        size: optional_json_string(body, "size")?,
        quality: optional_json_string(body, "quality")?,
        background: optional_json_string(body, "background")?,
        output_format: optional_json_string(body, "output_format")?,
        response_format: optional_json_string(body, "response_format")?,
        stream: optional_json_bool(body, "stream")?,
        partial_images: optional_json_usize(body, "partial_images")?,
        style: optional_json_string(body, "style")?,
        moderation: optional_json_string(body, "moderation")?,
        output_compression: optional_json_u8(body, "output_compression")?,
        user: optional_json_string(body, "user")?,
    };
    normalize_image_request(task, fields)
}

fn parse_raw_image_request(
    task: ImageTask,
    request: &HttpRequest,
    body: &[u8],
) -> Result<ParsedImageRequest, ApiError> {
    let query = request.query_string();
    let fields = ImageFields {
        model: query_param(query, "model"),
        prompt: query_param(query, "prompt"),
        images: if body.is_empty() {
            Vec::new()
        } else {
            vec![ParsedImageInput {
                filename: query_param(query, "filename").unwrap_or_else(|| "image.bin".to_string()),
                content: body.to_vec(),
            }]
        },
        mask: None,
        n: query_usize(query, "n")?,
        size: query_param(query, "size"),
        quality: query_param(query, "quality"),
        background: query_param(query, "background"),
        output_format: query_param(query, "output_format"),
        response_format: query_param(query, "response_format"),
        stream: query_bool(query, "stream")?,
        partial_images: query_usize(query, "partial_images")?,
        style: query_param(query, "style"),
        moderation: query_param(query, "moderation"),
        output_compression: query_u8(query, "output_compression")?,
        user: query_param(query, "user"),
    };
    normalize_image_request(task, fields)
}

fn parse_multipart_image_request(
    task: ImageTask,
    body: &[u8],
    boundary: &str,
) -> Result<ParsedImageRequest, ApiError> {
    let text = String::from_utf8_lossy(body);
    let marker = format!("--{boundary}");
    let mut fields = ImageFields::default();
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
            "image" => fields.images.push(ParsedImageInput {
                filename: disposition_param(disposition, "filename")
                    .unwrap_or_else(|| "image.bin".to_string()),
                content: value.as_bytes().to_vec(),
            }),
            "mask" => {
                fields.mask = Some(ParsedImageInput {
                    filename: disposition_param(disposition, "filename")
                        .unwrap_or_else(|| "mask.bin".to_string()),
                    content: value.as_bytes().to_vec(),
                });
            }
            "model" => fields.model = Some(trimmed_field(value)),
            "prompt" => fields.prompt = Some(trimmed_field(value)),
            "n" => fields.n = parse_usize_field("n", value)?,
            "size" => fields.size = Some(trimmed_field(value)),
            "quality" => fields.quality = Some(trimmed_field(value)),
            "background" => fields.background = Some(trimmed_field(value)),
            "output_format" => fields.output_format = Some(trimmed_field(value)),
            "response_format" => fields.response_format = Some(trimmed_field(value)),
            "stream" => fields.stream = parse_bool_field("stream", value)?,
            "partial_images" => fields.partial_images = parse_usize_field("partial_images", value)?,
            "style" => fields.style = Some(trimmed_field(value)),
            "moderation" => fields.moderation = Some(trimmed_field(value)),
            "output_compression" => {
                fields.output_compression = parse_u8_field("output_compression", value)?
            }
            "user" => fields.user = Some(trimmed_field(value)),
            _ => {}
        }
    }
    normalize_image_request(task, fields)
}

fn normalize_image_request(
    task: ImageTask,
    fields: ImageFields,
) -> Result<ParsedImageRequest, ApiError> {
    let model = fields
        .model
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_IMAGE_MODEL.to_string());
    let prompt = fields.prompt.filter(|value| !value.trim().is_empty());
    if task.requires_prompt() && prompt.is_none() {
        return Err(ApiError::bad_request(format!(
            "{} requires prompt",
            task.request_name()
        )));
    }
    if task.requires_image() && fields.images.is_empty() {
        return Err(ApiError::bad_request(format!(
            "{} requires image",
            task.request_name()
        )));
    }

    let n = fields.n.unwrap_or(1);
    if n == 0 || n > MAX_IMAGES_PER_REQUEST {
        return Err(ApiError::bad_request(format!(
            "n must be between 1 and {MAX_IMAGES_PER_REQUEST}"
        )));
    }
    let size = normalize_one_of(
        fields.size.as_deref().unwrap_or(DEFAULT_IMAGE_SIZE),
        "size",
        &[
            "auto",
            "256x256",
            "512x512",
            "1024x1024",
            "1024x1536",
            "1536x1024",
            "1792x1024",
            "1024x1792",
        ],
    )?;
    let quality = normalize_one_of(
        fields.quality.as_deref().unwrap_or(DEFAULT_IMAGE_QUALITY),
        "quality",
        &["auto", "low", "medium", "high", "standard", "hd"],
    )?;
    let background = normalize_one_of(
        fields.background.as_deref().unwrap_or("auto"),
        "background",
        &["auto", "transparent", "opaque"],
    )?;
    let output_format = normalize_one_of(
        fields
            .output_format
            .as_deref()
            .unwrap_or(DEFAULT_IMAGE_OUTPUT_FORMAT),
        "output_format",
        &["png", "jpeg", "webp"],
    )?;
    let response_format = normalize_response_format(fields.response_format.as_deref(), &model)?;
    let partial_images = fields
        .partial_images
        .unwrap_or(if fields.stream.unwrap_or(false) { 1 } else { 0 });
    if partial_images > MAX_PARTIAL_IMAGES {
        return Err(ApiError::bad_request(format!(
            "partial_images must be between 0 and {MAX_PARTIAL_IMAGES}"
        )));
    }
    if let Some(style) = fields.style.as_deref() {
        normalize_one_of(style, "style", &["vivid", "natural"])?;
    }
    if let Some(compression) = fields.output_compression
        && compression > 100
    {
        return Err(ApiError::bad_request(
            "output_compression must be between 0 and 100",
        ));
    }

    Ok(ParsedImageRequest {
        task,
        model,
        prompt,
        images: fields.images,
        mask: fields.mask,
        n,
        size,
        quality,
        background,
        output_format,
        response_format,
        stream: fields.stream.unwrap_or(false),
        partial_images,
        style: fields.style.filter(|value| !value.trim().is_empty()),
        moderation: fields.moderation.filter(|value| !value.trim().is_empty()),
        output_compression: fields.output_compression,
        user: fields.user.filter(|value| !value.trim().is_empty()),
    })
}

pub(crate) fn image_response_value(parsed: &ParsedImageRequest, created: u64) -> Value {
    let data = (0..parsed.n)
        .map(|index| image_object(parsed, index, 0))
        .collect::<Vec<_>>();
    let mut response = json!({
        "created": created,
        "data": data,
        "output_format": parsed.output_format,
        "quality": response_quality(parsed),
        "size": response_size(parsed),
        "usage": image_usage(parsed)
    });
    if parsed.background != "auto" {
        response["background"] = json!(parsed.background);
    }
    response
}

pub(crate) fn image_stream_frames(parsed: &ParsedImageRequest, created: u64) -> Vec<String> {
    let mut frames = Vec::new();
    let event_prefix = parsed.task.event_prefix();
    for partial_index in 0..parsed.partial_images {
        let event = format!("{event_prefix}.partial_image");
        let mut payload = image_event_payload(parsed, created, partial_index, 1);
        payload["type"] = json!(event);
        payload["partial_image_index"] = json!(partial_index);
        frames.push(sse_json_frame(&event, payload));
    }
    for index in 0..parsed.n {
        let event = format!("{event_prefix}.completed");
        let mut payload = image_event_payload(parsed, created, index, 0);
        payload["type"] = json!(event);
        frames.push(sse_json_frame(&event, payload));
    }
    frames.push("data: [DONE]\n\n".to_string());
    frames
}

fn image_stream_response(parsed: &ParsedImageRequest, created: u64) -> HttpResponse {
    let frames = image_stream_frames(parsed, created);
    HttpResponse::Ok()
        .insert_header(("cache-control", "no-cache"))
        .insert_header(("x-nerva-image-model", parsed.model.clone()))
        .insert_header(("x-nerva-image-backend", "deterministic-placeholder"))
        .content_type("text/event-stream")
        .streaming(stream::iter(frames.into_iter().map(|frame| {
            Ok::<web::Bytes, actix_web::Error>(web::Bytes::from(frame))
        })))
}

fn image_event_payload(
    parsed: &ParsedImageRequest,
    created: u64,
    index: usize,
    stage: u64,
) -> Value {
    let bytes = placeholder_png(parsed, index, stage);
    json!({
        "b64_json": image_base64(&bytes),
        "background": parsed.background,
        "created_at": created,
        "output_format": parsed.output_format,
        "quality": parsed.quality,
        "size": parsed.size,
        "usage": image_usage(parsed)
    })
}

fn image_object(parsed: &ParsedImageRequest, index: usize, stage: u64) -> Value {
    let bytes = placeholder_png(parsed, index, stage);
    let b64 = image_base64(&bytes);
    let mut object = match parsed.response_format {
        ImageResponseFormat::B64Json => json!({"b64_json": b64}),
        ImageResponseFormat::Url => json!({"url": format!("data:image/png;base64,{b64}")}),
    };
    if parsed.model == "dall-e-3"
        && let Some(prompt) = parsed.prompt.as_deref()
    {
        object["revised_prompt"] = json!(format!("NERVA placeholder image: {prompt}"));
    }
    object
}

fn placeholder_png(parsed: &ParsedImageRequest, index: usize, stage: u64) -> Vec<u8> {
    let seed = image_seed(parsed, index, stage);
    let (width, height) = placeholder_dimensions(&parsed.size);
    placeholder_png_bytes(seed, width, height, parsed.background == "transparent")
}

pub(crate) fn placeholder_png_bytes(
    seed: u64,
    width: u32,
    height: u32,
    transparent: bool,
) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(
        (height as usize).saturating_mul(1usize.saturating_add((width as usize).saturating_mul(4))),
    );
    let a = if transparent { 190 } else { 255 };
    for y in 0..height {
        pixels.push(0);
        for x in 0..width {
            let slot = seed
                .wrapping_add((x as u64).wrapping_mul(0x9e37_79b9))
                .wrapping_add((y as u64).wrapping_mul(0x85eb_ca6b));
            let checker = ((x / 8) ^ (y / 8)) & 1;
            pixels.push(((slot >> 16) as u8).wrapping_add((checker * 34) as u8));
            pixels.push(((slot >> 32) as u8).wrapping_add((checker * 21) as u8));
            pixels.push(((slot >> 48) as u8).wrapping_add((checker * 55) as u8));
            pixels.push(a);
        }
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(8);
    ihdr.push(6);
    ihdr.push(0);
    ihdr.push(0);
    ihdr.push(0);
    write_png_chunk(&mut out, b"IHDR", &ihdr);
    write_png_chunk(&mut out, b"IDAT", &zlib_store(&pixels));
    write_png_chunk(&mut out, b"IEND", &[]);
    out
}

pub(crate) fn image_base64(bytes: &[u8]) -> String {
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

pub(crate) fn image_base64_decode(value: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(value.len() * 3 / 4);
    let mut buffer = 0u32;
    let mut bits = 0u8;
    for byte in value.bytes() {
        if byte.is_ascii_whitespace() {
            continue;
        }
        if byte == b'=' {
            break;
        }
        let six = base64_value(byte)?;
        buffer = (buffer << 6) | u32::from(six);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buffer >> bits) & 0xff) as u8);
            if bits > 0 {
                buffer &= (1u32 << bits) - 1;
            } else {
                buffer = 0;
            }
        }
    }
    Some(out)
}

fn image_seed(parsed: &ParsedImageRequest, index: usize, stage: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    hash_bytes(&mut hash, parsed.task.seed_label().as_bytes());
    hash_bytes(&mut hash, parsed.model.as_bytes());
    hash_bytes(&mut hash, parsed.prompt.as_deref().unwrap_or("").as_bytes());
    hash_bytes(&mut hash, parsed.size.as_bytes());
    hash_bytes(&mut hash, parsed.quality.as_bytes());
    hash_bytes(&mut hash, parsed.background.as_bytes());
    hash_bytes(&mut hash, parsed.output_format.as_bytes());
    hash_bytes(&mut hash, parsed.style.as_deref().unwrap_or("").as_bytes());
    hash_bytes(
        &mut hash,
        parsed.moderation.as_deref().unwrap_or("").as_bytes(),
    );
    hash_bytes(&mut hash, parsed.user.as_deref().unwrap_or("").as_bytes());
    hash = hash.wrapping_mul(0x100_0000_01b3) ^ u64::from(parsed.output_compression.unwrap_or(0));
    for image in &parsed.images {
        hash_bytes(&mut hash, image.filename.as_bytes());
        hash_bytes(&mut hash, &image.content);
    }
    if let Some(mask) = parsed.mask.as_ref() {
        hash_bytes(&mut hash, mask.filename.as_bytes());
        hash_bytes(&mut hash, &mask.content);
    }
    hash ^ ((index as u64) << 32) ^ stage.rotate_left(17)
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    *hash ^= bytes.len() as u64;
    *hash = hash.wrapping_mul(0x100_0000_01b3);
    for byte in bytes.iter().take(4096) {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }
}

fn placeholder_dimensions(size: &str) -> (u32, u32) {
    match size {
        "1024x1536" | "1024x1792" => (64, 96),
        "1536x1024" | "1792x1024" => (96, 64),
        _ => (64, 64),
    }
}

fn response_quality(parsed: &ParsedImageRequest) -> &str {
    match parsed.quality.as_str() {
        "auto" | "standard" => "medium",
        "hd" => "high",
        quality => quality,
    }
}

fn response_size(parsed: &ParsedImageRequest) -> &str {
    if parsed.size == "auto" {
        DEFAULT_IMAGE_SIZE
    } else {
        &parsed.size
    }
}

fn image_usage(parsed: &ParsedImageRequest) -> Value {
    let text_tokens = parsed
        .prompt
        .as_deref()
        .map(estimated_text_tokens)
        .unwrap_or(0);
    let image_tokens = parsed.images.len().saturating_mul(85)
        + usize::from(parsed.mask.is_some()).saturating_mul(85);
    let output_tokens = parsed.n.saturating_mul(match parsed.quality.as_str() {
        "low" => 256,
        "high" | "hd" => 1536,
        _ => 1024,
    });
    json!({
        "input_tokens": text_tokens + image_tokens,
        "input_tokens_details": {
            "image_tokens": image_tokens,
            "text_tokens": text_tokens
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "image_tokens": output_tokens,
            "text_tokens": 0
        },
        "total_tokens": text_tokens + image_tokens + output_tokens
    })
}

fn estimated_text_tokens(text: &str) -> usize {
    text.split_whitespace()
        .count()
        .max(text.chars().count().div_ceil(4))
        .max(1)
}

fn write_png_chunk(out: &mut Vec<u8>, name: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(name);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(name.len() + data.len());
    crc_input.extend_from_slice(name);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len().saturating_add(16));
    out.extend_from_slice(&[0x78, 0x01]);
    for (index, chunk) in data.chunks(65_535).enumerate() {
        let final_block = usize::from(index + 1 == data.len().div_ceil(65_535)) as u8;
        out.push(final_block);
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in data {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    for byte in data {
        a = (a + u32::from(*byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
}

#[derive(Clone, Debug, Default)]
struct ImageFields {
    model: Option<String>,
    prompt: Option<String>,
    images: Vec<ParsedImageInput>,
    mask: Option<ParsedImageInput>,
    n: Option<usize>,
    size: Option<String>,
    quality: Option<String>,
    background: Option<String>,
    output_format: Option<String>,
    response_format: Option<String>,
    stream: Option<bool>,
    partial_images: Option<usize>,
    style: Option<String>,
    moderation: Option<String>,
    output_compression: Option<u8>,
    user: Option<String>,
}

impl ImageTask {
    fn requires_prompt(self) -> bool {
        matches!(self, ImageTask::Generation | ImageTask::Edit)
    }

    fn requires_image(self) -> bool {
        matches!(self, ImageTask::Edit | ImageTask::Variation)
    }

    fn request_name(self) -> &'static str {
        match self {
            ImageTask::Generation => "image generation",
            ImageTask::Edit => "image edit",
            ImageTask::Variation => "image variation",
        }
    }

    fn event_prefix(self) -> &'static str {
        match self {
            ImageTask::Generation => "image_generation",
            ImageTask::Edit => "image_edit",
            ImageTask::Variation => "image_variation",
        }
    }

    fn seed_label(self) -> &'static str {
        match self {
            ImageTask::Generation => "generation",
            ImageTask::Edit => "edit",
            ImageTask::Variation => "variation",
        }
    }
}

fn normalize_response_format(
    value: Option<&str>,
    model: &str,
) -> Result<ImageResponseFormat, ApiError> {
    let default = if model.starts_with("dall-e") {
        "url"
    } else {
        "b64_json"
    };
    match value.unwrap_or(default).trim() {
        "b64_json" => Ok(ImageResponseFormat::B64Json),
        "url" => Ok(ImageResponseFormat::Url),
        other => Err(ApiError::bad_request(format!(
            "response_format must be b64_json or url, got '{other}'"
        ))),
    }
}

fn normalize_one_of(
    value: &str,
    field: &'static str,
    allowed: &[&'static str],
) -> Result<String, ApiError> {
    let trimmed = value.trim();
    if allowed.contains(&trimmed) {
        Ok(trimmed.to_string())
    } else {
        Err(ApiError::bad_request(format!(
            "{field} must be one of {}",
            allowed.join(", ")
        )))
    }
}

fn optional_json_string(body: &Value, field: &'static str) -> Result<Option<String>, ApiError> {
    match body.get(field) {
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be a string"))),
    }
}

fn optional_json_bool(body: &Value, field: &'static str) -> Result<Option<bool>, ApiError> {
    match body.get(field) {
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{field} must be a boolean"))),
    }
}

fn optional_json_usize(body: &Value, field: &'static str) -> Result<Option<usize>, ApiError> {
    match body.get(field) {
        Some(Value::Number(number)) => number
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .map(Some)
            .ok_or_else(|| {
                ApiError::bad_request(format!("{field} must be a non-negative integer"))
            }),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!(
            "{field} must be a non-negative integer"
        ))),
    }
}

fn optional_json_u8(body: &Value, field: &'static str) -> Result<Option<u8>, ApiError> {
    match optional_json_usize(body, field)? {
        Some(value) => u8::try_from(value)
            .map(Some)
            .map_err(|_| ApiError::bad_request(format!("{field} must be between 0 and 255"))),
        None => Ok(None),
    }
}

fn json_image_inputs(value: Option<&Value>) -> Result<Vec<ParsedImageInput>, ApiError> {
    match value {
        Some(Value::Array(values)) => values
            .iter()
            .enumerate()
            .map(|(index, value)| json_image_input(Some(value), "image", index))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ApiError::bad_request("image array must not contain null")),
        Some(_) => json_image_input(value, "image", 0).map(|value| value.into_iter().collect()),
        None => Ok(Vec::new()),
    }
}

fn json_image_input(
    value: Option<&Value>,
    field: &'static str,
    index: usize,
) -> Result<Option<ParsedImageInput>, ApiError> {
    match value {
        Some(Value::String(value)) => Ok(Some(image_input_from_string(field, index, value))),
        Some(Value::Object(object)) => {
            let filename = object
                .get("filename")
                .and_then(Value::as_str)
                .unwrap_or(if field == "mask" {
                    "mask.png"
                } else {
                    "image.png"
                })
                .to_string();
            let data = object
                .get("content")
                .or_else(|| object.get("data"))
                .or_else(|| object.get("b64_json"))
                .or_else(|| object.get("url"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ApiError::bad_request(format!(
                        "{field} object requires content, data, b64_json, or url"
                    ))
                })?;
            Ok(Some(ParsedImageInput {
                filename,
                content: decode_image_string(data),
            }))
        }
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!(
            "{field} must be a string, object, or array"
        ))),
    }
}

fn image_input_from_string(field: &'static str, index: usize, value: &str) -> ParsedImageInput {
    let extension = if value.starts_with("data:image/") {
        value
            .split_once('/')
            .and_then(|(_, rest)| rest.split_once(';'))
            .map(|(extension, _)| extension)
            .unwrap_or("png")
    } else {
        "bin"
    };
    ParsedImageInput {
        filename: format!("{field}-{index}.{extension}"),
        content: decode_image_string(value),
    }
}

fn decode_image_string(value: &str) -> Vec<u8> {
    if let Some((_, encoded)) = value.split_once(";base64,") {
        return image_base64_decode(encoded).unwrap_or_else(|| value.as_bytes().to_vec());
    }
    image_base64_decode(value)
        .filter(|decoded| !decoded.is_empty())
        .unwrap_or_else(|| value.as_bytes().to_vec())
}

fn base64_value(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

fn parse_usize_field(field: &'static str, value: &str) -> Result<Option<usize>, ApiError> {
    value
        .trim()
        .parse::<usize>()
        .map(Some)
        .map_err(|_| ApiError::bad_request(format!("{field} must be a non-negative integer")))
}

fn parse_u8_field(field: &'static str, value: &str) -> Result<Option<u8>, ApiError> {
    value
        .trim()
        .parse::<u8>()
        .map(Some)
        .map_err(|_| ApiError::bad_request(format!("{field} must be between 0 and 255")))
}

fn parse_bool_field(field: &'static str, value: &str) -> Result<Option<bool>, ApiError> {
    match value.trim() {
        "true" | "1" => Ok(Some(true)),
        "false" | "0" => Ok(Some(false)),
        _ => Err(ApiError::bad_request(format!("{field} must be a boolean"))),
    }
}

fn query_usize(query: &str, name: &'static str) -> Result<Option<usize>, ApiError> {
    query_param(query, name)
        .as_deref()
        .map(|value| parse_usize_field(name, value))
        .transpose()
        .map(Option::flatten)
}

fn query_u8(query: &str, name: &'static str) -> Result<Option<u8>, ApiError> {
    query_param(query, name)
        .as_deref()
        .map(|value| parse_u8_field(name, value))
        .transpose()
        .map(Option::flatten)
}

fn query_bool(query: &str, name: &'static str) -> Result<Option<bool>, ApiError> {
    query_param(query, name)
        .as_deref()
        .map(|value| parse_bool_field(name, value))
        .transpose()
        .map(Option::flatten)
}

fn trimmed_field(value: &str) -> String {
    value.trim().to_string()
}

fn sse_json_frame(event: &str, value: Value) -> String {
    format!("event: {event}\ndata: {value}\n\n")
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
