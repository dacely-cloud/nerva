use std::collections::HashMap;
use std::fmt::Write as _;

use nerva_model::hf::tokenizer::PromptFormat;
use serde_json::{Map, Value, json};

use super::{
    ApiError, PromptInput, ReasoningMode, chat_messages_to_prompt, responses_input_to_prompt,
};

const BOS: &str = "<｜begin▁of▁sentence｜>";
const EOS: &str = "<｜end▁of▁sentence｜>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const DSML: &str = "｜DSML｜";
const REASONING_EFFORT_MAX: &str = concat!(
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n",
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n",
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n",
);

pub(crate) fn chat_prompt_for_reasoning(
    body: &Value,
    mode: ReasoningMode,
) -> Result<PromptInput, ApiError> {
    if mode == ReasoningMode::None {
        return Ok(PromptInput::Text {
            text: chat_messages_to_prompt(body)?,
            format: PromptFormat::Auto,
        });
    }
    Ok(PromptInput::Text {
        text: render_deepseek_chat_body(body, mode)?,
        format: PromptFormat::Raw,
    })
}

pub(crate) fn responses_prompt_for_reasoning(
    body: &Value,
    mode: ReasoningMode,
) -> Result<PromptInput, ApiError> {
    if mode == ReasoningMode::None {
        return Ok(PromptInput::Text {
            text: responses_input_to_prompt(body)?,
            format: PromptFormat::Auto,
        });
    }

    let mut messages = Vec::new();
    if let Some(instructions) = body.get("instructions").and_then(Value::as_str) {
        if !instructions.trim().is_empty() {
            messages.push(json!({"role": "system", "content": instructions}));
        }
    }
    match body.get("input") {
        Some(Value::String(input)) => messages.push(json!({"role": "user", "content": input})),
        Some(Value::Array(items)) => messages.extend(items.iter().cloned()),
        Some(_) => {
            return Err(ApiError::bad_request(
                "input must be a string or messages array",
            ));
        }
        None => return Err(ApiError::bad_request("responses require input")),
    }

    let mut rendered_body = json!({ "messages": messages });
    if let Some(tools) = body.get("tools") {
        rendered_body["tools"] = tools.clone();
    }
    if let Some(tool_choice) = body.get("tool_choice") {
        rendered_body["tool_choice"] = tool_choice.clone();
    }
    if let Some(reasoning_effort) = body.get("reasoning_effort") {
        rendered_body["reasoning_effort"] = reasoning_effort.clone();
    }
    if let Some(kwargs) = body.get("chat_template_kwargs") {
        rendered_body["chat_template_kwargs"] = kwargs.clone();
    }

    Ok(PromptInput::Text {
        text: render_deepseek_chat_body(&rendered_body, mode)?,
        format: PromptFormat::Raw,
    })
}

fn render_deepseek_chat_body(body: &Value, mode: ReasoningMode) -> Result<String, ApiError> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| ApiError::bad_request("chat completions require messages array"))?;
    if messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }

    let (thinking, max_reasoning_effort) = resolve_thinking(body, mode)?;
    let request_tools = request_tools(body);
    let synthetic_tool_system = !request_tools.is_empty()
        && !messages
            .iter()
            .any(|message| message.get("role").and_then(Value::as_str) == Some("system"));
    let rendered_tools_present = !request_tools.is_empty()
        || messages.iter().any(|message| {
            message.get("role").and_then(Value::as_str) == Some("developer")
                && message
                    .get("tools")
                    .and_then(Value::as_array)
                    .is_some_and(|tools| !tools.is_empty())
        });
    let drop_thinking = !rendered_tools_present;
    let last_user_render_index = last_user_render_index(messages, synthetic_tool_system);

    let mut out = String::from(BOS);
    if thinking && max_reasoning_effort {
        out.push_str(REASONING_EFFORT_MAX);
    }

    let mut tools_attached = false;
    let mut render_index = isize::from(synthetic_tool_system);
    if synthetic_tool_system {
        render_system_message(&mut out, None, request_tools)?;
        tools_attached = true;
    }

    for (message_index, message) in messages.iter().enumerate() {
        if is_following_tool_response(messages, message_index) {
            continue;
        }

        let current_render_index = render_index;
        render_index += 1;

        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user");
        match role {
            "system" => {
                let tools = if !tools_attached {
                    tools_attached = true;
                    request_tools
                } else {
                    &[]
                };
                let content = message_content(message)?;
                render_system_message(&mut out, Some(content.as_str()), tools)?;
            }
            "latest_reminder" => {
                out.push_str("<｜latest_reminder｜>");
                out.push_str(&message_content(message)?);
            }
            "developer" => render_developer_message(&mut out, message)?,
            "user" => render_user_message(&mut out, &message_content(message)?)?,
            "assistant" => {
                let emit_thinking =
                    thinking && (!drop_thinking || current_render_index > last_user_render_index);
                render_assistant_message(&mut out, message, emit_thinking)?;
            }
            "tool" => render_tool_response_block(&mut out, messages, message_index)?,
            _ => render_user_message(&mut out, &message_content(message)?)?,
        }

        if is_user_like(role) && next_rendered_entry_is_assistant_or_end(messages, message_index) {
            write_assistant_transition(
                &mut out,
                thinking,
                drop_thinking,
                current_render_index >= last_user_render_index,
            );
        }
    }

    Ok(out)
}

fn resolve_thinking(body: &Value, mode: ReasoningMode) -> Result<(bool, bool), ApiError> {
    let mut thinking = mode == ReasoningMode::DeepSeekThinking;
    let mut max_reasoning_effort = false;
    match body.get("reasoning_effort") {
        Some(Value::String(value)) if value == "none" => thinking = false,
        Some(Value::String(value)) if value == "max" || value == "xhigh" => {
            max_reasoning_effort = true;
        }
        Some(Value::String(_)) | Some(Value::Null) | None => {}
        Some(_) => return Err(ApiError::bad_request("reasoning_effort must be a string")),
    }
    Ok((thinking, max_reasoning_effort))
}

fn request_tools(body: &Value) -> &[Value] {
    if body.get("tool_choice") == Some(&Value::String("none".to_string())) {
        return &[];
    }
    body.get("tools")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice)
}

fn render_system_message(
    out: &mut String,
    content: Option<&str>,
    tools: &[Value],
) -> Result<(), ApiError> {
    if let Some(content) = content {
        out.push_str(content);
    }
    if !tools.is_empty() {
        out.push_str("\n\n");
        render_tools(out, tools)?;
    }
    Ok(())
}

fn render_developer_message(out: &mut String, message: &Value) -> Result<(), ApiError> {
    let content = message_content(message)?;
    if content.is_empty() {
        return Err(ApiError::bad_request(
            "invalid DeepSeek developer message: empty content",
        ));
    }
    out.push_str(USER);
    out.push_str(&content);
    if let Some(tools) = message.get("tools").and_then(Value::as_array)
        && !tools.is_empty()
    {
        out.push_str("\n\n");
        render_tools(out, tools)?;
    }
    Ok(())
}

fn render_user_message(out: &mut String, content: &str) -> Result<(), ApiError> {
    out.push_str(USER);
    out.push_str(content);
    Ok(())
}

fn render_tool_response_block(
    out: &mut String,
    messages: &[Value],
    message_index: usize,
) -> Result<(), ApiError> {
    let (start, end) = tool_response_block_bounds(messages, message_index);
    let mut indices = (start..end).collect::<Vec<_>>();
    if let Some(order) = last_tool_call_order_before(messages, start) {
        indices.sort_by_key(|index| {
            messages[*index]
                .get("tool_call_id")
                .and_then(Value::as_str)
                .and_then(|id| order.get(id))
                .copied()
                .unwrap_or(0)
        });
    }

    out.push_str(USER);
    for (offset, index) in indices.iter().enumerate() {
        if offset > 0 {
            out.push_str("\n\n");
        }
        out.push_str("<tool_result>");
        out.push_str(&message_content(&messages[*index])?);
        out.push_str("</tool_result>");
    }
    Ok(())
}

fn render_assistant_message(
    out: &mut String,
    message: &Value,
    emit_thinking: bool,
) -> Result<(), ApiError> {
    if emit_thinking {
        if let Some(reasoning) = message
            .get("reasoning_content")
            .or_else(|| message.get("reasoning"))
            .and_then(Value::as_str)
        {
            out.push_str(reasoning);
        }
        out.push_str(THINK_END);
    }

    let content = message_content(message)?;
    if message.get("task").and_then(Value::as_str) == Some("action") {
        out.push_str("<｜action｜>");
        out.push_str(&content);
    } else {
        out.push_str(&content);
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array)
        && !tool_calls.is_empty()
    {
        out.push_str("\n\n<｜DSML｜tool_calls>\n");
        for (index, tool_call) in tool_calls.iter().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            render_tool_call(out, tool_call)?;
        }
        out.push_str("\n</｜DSML｜tool_calls>");
    }

    out.push_str(EOS);
    Ok(())
}

fn render_tools(out: &mut String, tools: &[Value]) -> Result<(), ApiError> {
    out.push_str(
        r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:

<｜DSML｜tool_calls>
<｜DSML｜invoke name="$TOOL_NAME">
<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>
...
</｜DSML｜invoke>
<｜DSML｜invoke name="$TOOL_NAME2">
...
</｜DSML｜invoke>
</｜DSML｜tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

"#,
    );

    for (index, tool) in tools.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        render_tool_schema(out, tool)?;
    }
    out.push_str(
        "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n",
    );
    Ok(())
}

fn render_tool_schema(out: &mut String, tool: &Value) -> Result<(), ApiError> {
    let function = tool.get("function").unwrap_or(tool);
    let mut schema = Map::new();
    if let Some(name) = function.get("name") {
        schema.insert("name".to_string(), name.clone());
    }
    if let Some(description) = function.get("description") {
        schema.insert("description".to_string(), description.clone());
    }
    if let Some(parameters) = function.get("parameters") {
        schema.insert("parameters".to_string(), parameters.clone());
    }
    if let Some(strict) = function.get("strict").or_else(|| tool.get("strict")) {
        schema.insert("strict".to_string(), strict.clone());
    }
    out.push_str(&json_compact_spaced(&Value::Object(schema))?);
    Ok(())
}

fn render_tool_call(out: &mut String, tool_call: &Value) -> Result<(), ApiError> {
    let function = tool_call.get("function").unwrap_or(tool_call);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("assistant tool call requires function.name"))?;
    writeln!(out, "<{DSML}invoke name=\"{name}\">").expect("writing to String cannot fail");
    let arguments = function
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}");
    let arguments: Value = serde_json::from_str(arguments)
        .map_err(|err| ApiError::bad_request(format!("assistant tool call arguments: {err}")))?;
    let arguments = arguments.as_object().ok_or_else(|| {
        ApiError::bad_request("assistant tool call arguments must be a JSON object")
    })?;
    for (index, (key, value)) in arguments.iter().enumerate() {
        if index > 0 {
            out.push('\n');
        }
        let is_string = value.is_string();
        write!(
            out,
            "<{DSML}parameter name=\"{key}\" string=\"{}\">",
            if is_string { "true" } else { "false" }
        )
        .expect("writing to String cannot fail");
        match value {
            Value::String(value) => out.push_str(value),
            value => out.push_str(&json_compact_spaced(value)?),
        }
        write!(out, "</{DSML}parameter>").expect("writing to String cannot fail");
    }
    write!(out, "\n</{DSML}invoke>").expect("writing to String cannot fail");
    Ok(())
}

fn message_content(message: &Value) -> Result<String, ApiError> {
    match message.get("content") {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    out.push_str(text);
                }
            }
            Ok(out)
        }
        Some(Value::Null) | None => Ok(String::new()),
        Some(_) => Err(ApiError::bad_request("message content must be text")),
    }
}

fn write_assistant_transition(
    out: &mut String,
    thinking: bool,
    drop_thinking: bool,
    opens_thinking: bool,
) {
    out.push_str(ASSISTANT);
    if thinking && (!drop_thinking || opens_thinking) {
        out.push_str(THINK_START);
    } else {
        out.push_str(THINK_END);
    }
}

fn last_user_render_index(messages: &[Value], synthetic_tool_system: bool) -> isize {
    let mut render_index = isize::from(synthetic_tool_system);
    let mut last = -1;
    for (index, message) in messages.iter().enumerate() {
        if is_following_tool_response(messages, index) {
            continue;
        }
        if is_user_like(
            message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user"),
        ) {
            last = render_index;
        }
        render_index += 1;
    }
    last
}

fn is_user_like(role: &str) -> bool {
    matches!(role, "developer" | "user" | "tool")
}

fn is_following_tool_response(messages: &[Value], index: usize) -> bool {
    messages[index].get("role").and_then(Value::as_str) == Some("tool")
        && index > 0
        && messages[index - 1].get("role").and_then(Value::as_str) == Some("tool")
}

fn next_rendered_entry_is_assistant_or_end(messages: &[Value], index: usize) -> bool {
    let current_role = messages[index].get("role").and_then(Value::as_str);
    let mut next = index + 1;
    if current_role == Some("tool") {
        while next < messages.len()
            && messages[next].get("role").and_then(Value::as_str) == Some("tool")
        {
            next += 1;
        }
    }
    messages
        .get(next)
        .and_then(|message| message.get("role").and_then(Value::as_str))
        .map(|role| role == "assistant")
        .unwrap_or(true)
}

fn tool_response_block_bounds(messages: &[Value], index: usize) -> (usize, usize) {
    let mut start = index;
    while start > 0 && messages[start - 1].get("role").and_then(Value::as_str) == Some("tool") {
        start -= 1;
    }
    let mut end = index + 1;
    while end < messages.len() && messages[end].get("role").and_then(Value::as_str) == Some("tool")
    {
        end += 1;
    }
    (start, end)
}

fn last_tool_call_order_before(messages: &[Value], index: usize) -> Option<HashMap<&str, usize>> {
    let mut order = None;
    for message in &messages[..index] {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        let current = tool_calls
            .iter()
            .enumerate()
            .filter_map(|(index, call)| {
                call.get("id").and_then(Value::as_str).map(|id| (id, index))
            })
            .collect::<HashMap<_, _>>();
        if !current.is_empty() {
            order = Some(current);
        }
    }
    order
}

fn json_compact_spaced(value: &Value) -> Result<String, ApiError> {
    match value {
        Value::Null => Ok("null".to_string()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Number(value) => Ok(value.to_string()),
        Value::String(value) => serde_json::to_string(value)
            .map_err(|err| ApiError::internal(format!("json encode failed: {err}"))),
        Value::Array(values) => {
            let mut out = String::from("[");
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(&json_compact_spaced(value)?);
            }
            out.push(']');
            Ok(out)
        }
        Value::Object(map) => {
            let mut out = String::from("{");
            for (index, (key, value)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(
                    &serde_json::to_string(key)
                        .map_err(|err| ApiError::internal(format!("json encode failed: {err}")))?,
                );
                out.push_str(": ");
                out.push_str(&json_compact_spaced(value)?);
            }
            out.push('}');
            Ok(out)
        }
    }
}
