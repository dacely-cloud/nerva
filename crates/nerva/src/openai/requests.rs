use actix_web::HttpRequest;
use nerva_core::types::id::token::TokenId;
use nerva_model::causal_lm::types::HfCausalLmStopReason;
use nerva_model::hf::tokenizer::{PromptFormat, decode_generated_text};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, GeneratedText, PromptInput, ReasoningMode, validate_functions_field,
    validate_mcp_tools_field, validate_tool_choice_field,
};

pub(crate) fn authorize(state: &AppState, request: &HttpRequest) -> Result<(), ApiError> {
    let Some(api_key) = state.config.api_key.as_deref() else {
        return Ok(());
    };
    let Some(value) = request.headers().get("authorization") else {
        return Err(ApiError::unauthorized());
    };
    let Ok(value) = value.to_str() else {
        return Err(ApiError::unauthorized());
    };
    if value == format!("Bearer {api_key}") {
        Ok(())
    } else {
        Err(ApiError::unauthorized())
    }
}

pub(crate) fn require_known_model(state: &AppState, body: &Value) -> Result<(), ApiError> {
    let Some(model) = body.get("model").and_then(Value::as_str) else {
        return Ok(());
    };
    if model == state.config.model_id {
        Ok(())
    } else {
        Err(ApiError::not_found(format!(
            "model '{model}' is not served by this NERVA instance"
        )))
    }
}

pub(crate) fn reject_unsupported_generation_options(body: &Value) -> Result<(), ApiError> {
    reject_unsupported_generation_options_common(body)?;
    reject_nonempty_field(body, "tools")?;
    reject_nonempty_field(body, "functions")?;
    if let Some(tool_choice) = body.get("tool_choice") {
        if tool_choice != "none" {
            return Err(ApiError::unsupported(
                "tool_choice requires the chat or responses MCP execution path",
            ));
        }
    }
    reject_unsupported_response_format(body)
}

pub(crate) fn reject_unsupported_generation_options_with_tools(
    body: &Value,
) -> Result<(), ApiError> {
    reject_unsupported_generation_options_common(body)?;
    reject_present_field(body, "echo")?;
    reject_present_field(body, "suffix")?;
    validate_functions_field(body)?;
    validate_mcp_tools_field(body)?;
    validate_tool_choice_field(body)?;
    reject_unsupported_response_format(body)
}

fn reject_unsupported_generation_options_common(body: &Value) -> Result<(), ApiError> {
    reject_nonzero_penalty(body, "presence_penalty")?;
    reject_nonzero_penalty(body, "frequency_penalty")?;
    reject_nonempty_field(body, "logit_bias")?;
    reject_present_field(body, "logprobs")?;
    reject_present_field(body, "top_logprobs")?;
    reject_present_field(body, "best_of")?;
    Ok(())
}

fn reject_unsupported_response_format(body: &Value) -> Result<(), ApiError> {
    if let Some(response_format) = body.get("response_format") {
        let ty = response_format
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("text");
        if !matches!(ty, "text" | "json_object" | "json_schema") {
            return Err(ApiError::unsupported(
                "response_format type must be text, json_object, or json_schema",
            ));
        }
    }
    Ok(())
}

fn reject_present_field(body: &Value, name: &'static str) -> Result<(), ApiError> {
    if body.get(name).is_some_and(|value| !value.is_null()) {
        Err(ApiError::unsupported(format!(
            "{name} is not implemented by the NERVA OpenAI server yet"
        )))
    } else {
        Ok(())
    }
}

fn reject_nonempty_field(body: &Value, name: &'static str) -> Result<(), ApiError> {
    let Some(value) = body.get(name) else {
        return Ok(());
    };
    let empty = match value {
        Value::Null => true,
        Value::Array(items) => items.is_empty(),
        Value::Object(items) => items.is_empty(),
        _ => false,
    };
    if empty {
        Ok(())
    } else {
        Err(ApiError::unsupported(format!(
            "{name} is not implemented by the NERVA OpenAI server yet"
        )))
    }
}

fn reject_nonzero_penalty(body: &Value, name: &'static str) -> Result<(), ApiError> {
    let Some(value) = body.get(name).and_then(Value::as_f64) else {
        return Ok(());
    };
    if value == 0.0 {
        Ok(())
    } else {
        Err(ApiError::unsupported(format!(
            "{name} is not implemented by the NERVA OpenAI server yet"
        )))
    }
}

pub(crate) fn request_n(body: &Value) -> Result<usize, ApiError> {
    let value = body
        .get("n")
        .and_then(Value::as_u64)
        .map(|value| usize::try_from(value).map_err(|_| ApiError::bad_request("n is too large")))
        .transpose()?
        .unwrap_or(1);
    if value == 0 {
        Err(ApiError::bad_request("n must be non-zero"))
    } else {
        Ok(value)
    }
}

pub(crate) fn request_stream(body: &Value) -> bool {
    body.get("stream").and_then(Value::as_bool).unwrap_or(false)
}

pub(crate) fn request_echo(body: &Value) -> Result<bool, ApiError> {
    request_bool(body, "echo", false)
}

pub(crate) fn request_include_reasoning(body: &Value) -> Result<bool, ApiError> {
    request_bool(body, "include_reasoning", true)
}

pub(crate) fn request_reasoning_mode(
    state: &AppState,
    body: &Value,
) -> Result<ReasoningMode, ApiError> {
    if !served_model_is_deepseek(state) {
        return Ok(ReasoningMode::None);
    }
    let thinking = request_deepseek_thinking(body)?;
    if thinking {
        Ok(ReasoningMode::DeepSeekThinking)
    } else {
        Ok(ReasoningMode::DeepSeekChat)
    }
}

pub(crate) const fn prompt_format_for_reasoning(mode: ReasoningMode) -> PromptFormat {
    match mode {
        ReasoningMode::None => PromptFormat::Auto,
        ReasoningMode::DeepSeekChat => PromptFormat::DeepSeekChat,
        ReasoningMode::DeepSeekThinking => PromptFormat::DeepSeekThinking,
    }
}

fn served_model_is_deepseek(state: &AppState) -> bool {
    let model_id = state.config.model_id.to_ascii_lowercase();
    let model_path = state.config.model_path.to_ascii_lowercase();
    model_id.contains("deepseek") || model_path.contains("deepseek")
}

fn request_deepseek_thinking(body: &Value) -> Result<bool, ApiError> {
    let thinking = request_template_bool(body, "thinking")?;
    let enable_thinking = request_template_bool(body, "enable_thinking")?;
    if let (Some(thinking), Some(enable_thinking)) = (thinking, enable_thinking) {
        if thinking != enable_thinking {
            return Err(ApiError::bad_request(
                "thinking and enable_thinking must match when both are set",
            ));
        }
    }
    let template_thinking = thinking.or(enable_thinking).unwrap_or(false);
    if template_thinking {
        return Ok(true);
    }
    match body.get("reasoning_effort") {
        Some(Value::String(effort)) => Ok(effort != "none"),
        Some(Value::Null) | None => Ok(false),
        Some(_) => Err(ApiError::bad_request("reasoning_effort must be a string")),
    }
}

fn request_template_bool(body: &Value, name: &'static str) -> Result<Option<bool>, ApiError> {
    let top_level = match body.get(name) {
        Some(Value::Bool(value)) => Some(*value),
        Some(Value::Null) | None => None,
        Some(_) => return Err(ApiError::bad_request(format!("{name} must be a boolean"))),
    };
    let template = match body
        .get("chat_template_kwargs")
        .and_then(|value| value.get(name))
    {
        Some(Value::Bool(value)) => Some(*value),
        Some(Value::Null) | None => None,
        Some(_) => {
            return Err(ApiError::bad_request(format!(
                "chat_template_kwargs.{name} must be a boolean"
            )));
        }
    };
    if let (Some(top_level), Some(template)) = (top_level, template) {
        if top_level != template {
            return Err(ApiError::bad_request(format!(
                "{name} and chat_template_kwargs.{name} must match when both are set"
            )));
        }
    }
    Ok(top_level.or(template))
}

fn request_bool(body: &Value, name: &'static str, default: bool) -> Result<bool, ApiError> {
    match body.get(name) {
        Some(Value::Bool(value)) => Ok(*value),
        Some(Value::Null) | None => Ok(default),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be a boolean"))),
    }
}

pub(crate) fn completion_prompts(body: &Value) -> Result<Vec<PromptInput>, ApiError> {
    match body.get("prompt") {
        Some(Value::String(prompt)) => Ok(vec![text_prompt(prompt.clone(), PromptFormat::Raw)]),
        Some(Value::Array(items)) if items.iter().all(Value::is_number) => {
            Ok(vec![PromptInput::TokenIds(parse_token_id_array(items)?)])
        }
        Some(Value::Array(items)) if items.iter().all(Value::is_string) => items
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(|prompt| text_prompt(prompt.to_string(), PromptFormat::Raw))
                    .ok_or_else(|| ApiError::bad_request("prompt arrays must contain strings"))
            })
            .collect(),
        Some(Value::Array(items)) if items.iter().all(Value::is_array) => items
            .iter()
            .map(|value| {
                let tokens = value.as_array().ok_or_else(|| {
                    ApiError::bad_request("prompt token arrays must contain arrays")
                })?;
                Ok(PromptInput::TokenIds(parse_token_id_array(tokens)?))
            })
            .collect(),
        Some(_) => Err(ApiError::bad_request(
            "prompt must be a string, token-id array, string array, or token-id array array",
        )),
        None => Err(ApiError::bad_request("missing prompt")),
    }
}

fn text_prompt(text: String, format: PromptFormat) -> PromptInput {
    PromptInput::Text { text, format }
}

pub(crate) fn empty_text_prompt() -> PromptInput {
    text_prompt(String::new(), PromptFormat::Raw)
}

fn parse_token_id_array(items: &[Value]) -> Result<Vec<TokenId>, ApiError> {
    if items.is_empty() {
        return Err(ApiError::bad_request(
            "prompt token arrays must not be empty",
        ));
    }
    items
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|token| u32::try_from(token).ok())
                .map(TokenId)
                .ok_or_else(|| ApiError::bad_request("prompt token ids must fit u32"))
        })
        .collect()
}

pub(crate) fn chat_messages_to_prompt(body: &Value) -> Result<String, ApiError> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| ApiError::bad_request("chat completions require messages array"))?;
    if messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    let mut prompt = String::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        let content = message_content_text(message.get("content"))?;
        if content.trim().is_empty() {
            continue;
        }
        match role {
            "system" => prompt.push_str("System: "),
            "assistant" => prompt.push_str("Assistant: "),
            "tool" => prompt.push_str("Tool: "),
            _ => prompt.push_str("User: "),
        }
        prompt.push_str(&content);
        prompt.push('\n');
    }
    prompt.push_str("Assistant:");
    Ok(prompt)
}

pub(crate) fn responses_input_to_prompt(body: &Value) -> Result<String, ApiError> {
    let mut prompt = String::new();
    if let Some(instructions) = body.get("instructions").and_then(Value::as_str) {
        if !instructions.trim().is_empty() {
            prompt.push_str("System: ");
            prompt.push_str(instructions);
            prompt.push('\n');
        }
    }
    match body.get("input") {
        Some(Value::String(input)) => prompt.push_str(input),
        Some(Value::Array(messages)) => {
            let body = json!({ "messages": messages });
            prompt.push_str(&chat_messages_to_prompt(&body)?);
        }
        Some(_) => {
            return Err(ApiError::bad_request(
                "input must be a string or messages array",
            ));
        }
        None => return Err(ApiError::bad_request("responses require input")),
    }
    Ok(prompt)
}

fn message_content_text(content: Option<&Value>) -> Result<String, ApiError> {
    match content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") | Some("input_text") | None => {
                        if let Some(value) = part.get("text").and_then(Value::as_str) {
                            if !text.is_empty() {
                                text.push('\n');
                            }
                            text.push_str(value);
                        }
                    }
                    Some(other) => {
                        return Err(ApiError::unsupported(format!(
                            "message content part '{other}' is not supported by the text-only backend"
                        )));
                    }
                }
            }
            Ok(text)
        }
        Some(Value::Null) | None => Ok(String::new()),
        Some(_) => Err(ApiError::bad_request(
            "message content must be a string or text content array",
        )),
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ReasoningSplit {
    pub(crate) reasoning: String,
    pub(crate) content: String,
}

pub(crate) fn split_generated_reasoning(text: &str, mode: ReasoningMode) -> ReasoningSplit {
    const THINK_START: &str = "<think>";
    const THINK_END: &str = "</think>";

    if mode == ReasoningMode::None {
        return ReasoningSplit {
            reasoning: String::new(),
            content: text.to_string(),
        };
    }

    let assume_prefix_reasoning = mode == ReasoningMode::DeepSeekThinking;
    if let Some(start_index) = text.find(THINK_START) {
        let before = &text[..start_index];
        let after_start = &text[start_index + THINK_START.len()..];
        if let Some(end_index) = after_start.find(THINK_END) {
            let after_end = &after_start[end_index + THINK_END.len()..];
            return ReasoningSplit {
                reasoning: after_start[..end_index].to_string(),
                content: format!("{before}{after_end}"),
            };
        }
        return ReasoningSplit {
            reasoning: after_start.to_string(),
            content: before.to_string(),
        };
    }

    if let Some(end_index) = text.find(THINK_END) {
        let after_end = &text[end_index + THINK_END.len()..];
        return if assume_prefix_reasoning {
            ReasoningSplit {
                reasoning: text[..end_index].to_string(),
                content: after_end.to_string(),
            }
        } else {
            ReasoningSplit {
                reasoning: String::new(),
                content: format!("{}{}", &text[..end_index], after_end),
            }
        };
    }

    if assume_prefix_reasoning {
        ReasoningSplit {
            reasoning: text.to_string(),
            content: String::new(),
        }
    } else {
        ReasoningSplit {
            reasoning: String::new(),
            content: text.to_string(),
        }
    }
}

pub(crate) fn request_max_tokens(state: &AppState, body: &Value) -> Result<usize, ApiError> {
    let value = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(Value::as_u64)
        .map(|value| {
            usize::try_from(value).map_err(|_| ApiError::bad_request("max_tokens is too large"))
        })
        .transpose()?
        .unwrap_or(state.config.default_output_tokens);
    if value == 0 {
        Err(ApiError::bad_request("max_tokens must be non-zero"))
    } else {
        Ok(value)
    }
}

pub(crate) fn request_f32(body: &Value, name: &'static str, default: f32) -> Result<f32, ApiError> {
    let value = body
        .get(name)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .unwrap_or(default);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ApiError::bad_request(format!("{name} must be finite")))
    }
}

pub(crate) fn request_u32(body: &Value, name: &'static str, default: u32) -> Result<u32, ApiError> {
    body.get(name)
        .and_then(Value::as_u64)
        .map(|value| {
            u32::try_from(value).map_err(|_| ApiError::bad_request(format!("{name} is too large")))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

pub(crate) fn request_u64_opt(body: &Value, name: &'static str) -> Result<Option<u64>, ApiError> {
    match body.get(name) {
        Some(Value::Number(number)) => number
            .as_u64()
            .map(Some)
            .ok_or_else(|| ApiError::bad_request(format!("{name} must be a u64"))),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be a u64"))),
    }
}

pub(crate) fn request_stop_strings(body: &Value) -> Result<Vec<String>, ApiError> {
    match body.get("stop") {
        Some(Value::String(stop)) => Ok(vec![stop.clone()]),
        Some(Value::Array(stops)) => stops
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_string)
                    .ok_or_else(|| ApiError::bad_request("stop array must contain strings"))
            })
            .collect(),
        Some(Value::Null) | None => Ok(Vec::new()),
        Some(_) => Err(ApiError::bad_request(
            "stop must be a string or an array of strings",
        )),
    }
}

pub(crate) fn request_optional_string(
    body: &Value,
    name: &'static str,
) -> Result<Option<String>, ApiError> {
    match body.get(name) {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.clone())),
        Some(Value::String(_)) => Err(ApiError::bad_request(format!("{name} must not be empty"))),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be a string"))),
    }
}

pub(crate) fn request_suffix(body: &Value) -> Result<Option<String>, ApiError> {
    match body.get("suffix") {
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request("suffix must be a string")),
    }
}

pub(crate) fn string_any(body: &Value, names: &[&str]) -> Result<Option<String>, ApiError> {
    for name in names {
        if let Some(value) = body.get(*name) {
            return value
                .as_str()
                .map(|value| Some(value.to_string()))
                .ok_or_else(|| ApiError::bad_request(format!("{name} must be a string")));
        }
    }
    Ok(None)
}

pub(crate) fn completion_echo_prefix(
    model_path: &str,
    prompt: &PromptInput,
    echo: bool,
) -> Result<Option<String>, ApiError> {
    if !echo {
        return Ok(None);
    }
    match prompt {
        PromptInput::Text { text, .. } => Ok(Some(text.clone())),
        PromptInput::TokenIds(tokens) => decode_generated_text(model_path, tokens)
            .map_err(ApiError::bad_request)?
            .ok_or_else(|| ApiError::bad_request("tokenizer decode is unavailable"))
            .map(Some),
    }
}

pub(crate) fn request_response_format_instruction(
    body: &Value,
) -> Result<Option<String>, ApiError> {
    let Some(response_format) = body.get("response_format") else {
        return Ok(None);
    };
    if response_format.is_null() {
        return Ok(None);
    }
    let ty = response_format
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("text");
    match ty {
        "text" => Ok(None),
        "json_object" => Ok(Some(
            "Respond with a single valid JSON object and no surrounding prose.".to_string(),
        )),
        "json_schema" => {
            let json_schema = response_format.get("json_schema").ok_or_else(|| {
                ApiError::bad_request("response_format json_schema requires json_schema")
            })?;
            let name = json_schema
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("response");
            let schema = json_schema.get("schema").unwrap_or(json_schema);
            Ok(Some(format!(
                "Respond with JSON matching schema '{name}': {}",
                schema
            )))
        }
        _ => Err(ApiError::unsupported(
            "response_format type must be text, json_object, or json_schema",
        )),
    }
}

pub(crate) fn apply_response_format_instruction(
    prompt: PromptInput,
    instruction: Option<&str>,
) -> Result<PromptInput, ApiError> {
    let Some(instruction) = instruction else {
        return Ok(prompt);
    };
    match prompt {
        PromptInput::Text { text, format } => Ok(PromptInput::Text {
            text: append_assistant_instruction(text, instruction),
            format,
        }),
        PromptInput::TokenIds(_) => Err(ApiError::unsupported(
            "response_format with token-id prompts requires a text prompt",
        )),
    }
}

pub(crate) fn append_assistant_instruction(mut prompt: String, instruction: &str) -> String {
    const ASSISTANT_MARKER: &str = "Assistant:";
    if prompt.ends_with(ASSISTANT_MARKER) {
        let new_len = prompt.len().saturating_sub(ASSISTANT_MARKER.len());
        prompt.truncate(new_len);
        prompt.push_str("System: ");
        prompt.push_str(instruction);
        prompt.push('\n');
        prompt.push_str(ASSISTANT_MARKER);
    } else {
        if !prompt.ends_with('\n') {
            prompt.push('\n');
        }
        prompt.push_str(instruction);
    }
    prompt
}

pub(crate) fn completion_text(
    generated: &str,
    output_prefix: Option<&str>,
    output_suffix: Option<&str>,
) -> String {
    let mut text = String::new();
    if let Some(prefix) = output_prefix {
        text.push_str(prefix);
    }
    text.push_str(generated);
    if let Some(suffix) = output_suffix {
        text.push_str(suffix);
    }
    text
}

pub(crate) fn validate_sampling(temperature: f32, top_p: f32) -> Result<(), ApiError> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(ApiError::bad_request("temperature must be finite and >= 0"));
    }
    if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
        return Err(ApiError::bad_request("top_p must be finite and in (0, 1]"));
    }
    Ok(())
}

pub(crate) fn apply_stop_strings(text: String, stops: &[String]) -> (String, bool) {
    let Some(index) = stops
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()
    else {
        return (text, false);
    };
    (text[..index].to_string(), true)
}

pub(crate) fn finish_reason(
    stop_reason: HfCausalLmStopReason,
    stopped_by_stop_string: bool,
) -> &'static str {
    if stopped_by_stop_string || stop_reason == HfCausalLmStopReason::EosToken {
        "stop"
    } else {
        "length"
    }
}

pub(crate) fn usage(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    })
}

pub(crate) fn generated_metadata(generated: &GeneratedText) -> Value {
    json!({
        "cache_key": &generated.cache_key,
        "cache_hit": generated.cache_hit,
        "session_id": &generated.session_id,
        "prompt_hash": format!("{:016x}", generated.prompt_hash)
    })
}
