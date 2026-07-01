use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, McpApprovalRequest, McpServerRecord, McpToolExecution, McpToolInvocation,
    McpToolResult, authorize, unix_seconds,
};

const MCP_PROTOCOL_VERSION: &str = "2025-06-18";
const DEFAULT_MCP_READ_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MCP_CONNECT_TIMEOUT_SECS: u64 = 10;

pub(crate) async fn register_mcp_server(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let connector_id = optional_string(&body, "connector_id")?;
        let command = optional_string(&body, "command")?;
        let endpoint = optional_string(&body, "endpoint")?
            .or(optional_string(&body, "server_url")?)
            .or(optional_string(&body, "url")?);
        let transport = optional_string(&body, "transport")?
            .unwrap_or_else(|| infer_mcp_transport(endpoint.as_deref(), command.as_deref(), connector_id.as_deref()));
        let endpoint = endpoint.unwrap_or_else(|| {
            connector_id
                .as_ref()
                .map(|id| format!("connector:{id}"))
                .unwrap_or_default()
        });
        if endpoint.is_empty() && command.is_none() {
            return Err(ApiError::bad_request(
                "MCP server registration requires endpoint, server_url, url, connector_id, or command",
            ));
        }
        let id = optional_string(&body, "id")?
            .or(optional_string(&body, "server_label")?)
            .unwrap_or_else(|| state.next_response_id("mcp"));
        let protocol_version = optional_string(&body, "protocol_version")?
            .unwrap_or_else(|| MCP_PROTOCOL_VERSION.to_string());
        let probe = body.get("probe").and_then(Value::as_bool).unwrap_or(true);
        let now = unix_seconds();
        let mut record = McpServerRecord {
            id: id.clone(),
            created: now,
            updated: now,
            transport,
            endpoint,
            message_endpoint: optional_string(&body, "message_endpoint")?,
            command,
            args: string_array(&body, "args")?,
            cwd: optional_string(&body, "cwd")?,
            env: string_map(&body, "env")?,
            authorization: optional_string(&body, "authorization")?,
            protocol_version,
            session_id: optional_string(&body, "session_id")?,
            connector_id,
            allowed_tools: string_array(&body, "allowed_tools")?,
            require_approval: body
                .get("require_approval")
                .cloned()
                .unwrap_or_else(|| json!("always")),
            status: "registered".to_string(),
            capabilities: Value::Null,
            tools: json!([]),
            last_error: None,
        };
        if probe {
            let probe = probe_mcp_server(&record)?;
            record.status = "connected".to_string();
            record.protocol_version = probe.protocol_version;
            record.session_id = probe.session_id;
            record.message_endpoint = probe.message_endpoint;
            record.capabilities = probe.capabilities;
            record.tools = filter_mcp_tools(probe.tools, &record.allowed_tools);
            record.updated = unix_seconds();
        }
        lock_mcp_servers(&state)?.insert(id, record.clone());
        Ok::<_, ApiError>(HttpResponse::Ok().json(mcp_server_json(&record)))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_mcp_servers(
    state: web::Data<AppState>,
    request: HttpRequest,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let servers = lock_mcp_servers(&state)?
            .values()
            .map(mcp_server_json)
            .collect::<Vec<_>>();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "list",
            "data": servers
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn list_mcp_server_tools(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let servers = lock_mcp_servers(&state)?;
        let record = servers
            .get(&id)
            .ok_or_else(|| ApiError::not_found(format!("MCP server '{id}' does not exist")))?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "list",
            "data": record.tools
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn call_mcp_server_tool(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
    body: web::Json<Value>,
) -> HttpResponse {
    let id = path.into_inner();
    call_mcp_tool_by_id(state, request, id, body).await
}

pub(crate) async fn call_mcp_tool(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let id = match body.get("server_id").and_then(Value::as_str) {
        Some(id) => id.to_string(),
        None => {
            return ApiError::bad_request("MCP tool call requires server_id").into_response();
        }
    };
    call_mcp_tool_by_id(state, request, id, body).await
}

async fn call_mcp_tool_by_id(
    state: web::Data<AppState>,
    request: HttpRequest,
    id: String,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let name = body
            .get("name")
            .or_else(|| body.get("tool"))
            .and_then(Value::as_str)
            .ok_or_else(|| ApiError::bad_request("MCP tool call requires name"))?;
        let arguments = body
            .get("arguments")
            .or_else(|| body.get("args"))
            .cloned()
            .unwrap_or_else(|| json!({}));
        let mut record = {
            let servers = lock_mcp_servers(&state)?;
            servers
                .get(&id)
                .cloned()
                .ok_or_else(|| ApiError::not_found(format!("MCP server '{id}' does not exist")))?
        };
        if let Some(authorization) = optional_string(&body, "authorization")? {
            record.authorization = Some(authorization);
        }
        let result = call_mcp_tool_sync(&record, name, arguments.clone())?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "server_id": id,
            "server_label": record.id,
            "tool": name,
            "arguments": arguments,
            "result": result
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn execute_request_mcp_tool(
    state: web::Data<AppState>,
    body: &Value,
) -> Result<Option<McpToolExecution>, ApiError> {
    let Some(invocation) = mcp_tool_invocation_from_request(body)? else {
        return Ok(None);
    };
    let record = resolve_mcp_server_for_invocation(&state, &invocation)?;
    validate_mcp_tool_allowed(&record.allowed_tools, &invocation.name)?;
    if mcp_tool_needs_approval(&invocation.require_approval, &invocation.name)
        && !request_contains_mcp_approval(body)?
    {
        return Ok(Some(McpToolExecution::ApprovalRequired(
            mcp_approval_request(&state, &record, &invocation),
        )));
    }
    let server_id = record.id.clone();
    let server_label = invocation
        .server_label
        .clone()
        .or_else(|| Some(record.id.clone()));
    let name = invocation.name.clone();
    let arguments = invocation.arguments.clone();
    let result_name = name.clone();
    let result_arguments = arguments.clone();
    let result = web::block(move || call_mcp_tool_sync(&record, &name, arguments))
        .await
        .map_err(|err| ApiError::internal(format!("MCP tool task failed: {err}")))??;
    Ok(Some(McpToolExecution::Completed(McpToolResult {
        server_id,
        server_label,
        name: result_name,
        arguments: result_arguments,
        result,
    })))
}

pub(crate) async fn delete_mcp_server(
    state: web::Data<AppState>,
    request: HttpRequest,
    path: web::Path<String>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = path.into_inner();
        let removed = lock_mcp_servers(&state)?.remove(&id).is_some();
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "mcp_server.deleted",
            "deleted": removed
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) fn validate_mcp_tools_field(body: &Value) -> Result<(), ApiError> {
    let Some(tools) = body.get("tools") else {
        return Ok(());
    };
    let Value::Array(tools) = tools else {
        return Err(ApiError::bad_request("tools must be an array"));
    };
    for tool in tools {
        let ty = tool
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("function");
        match ty {
            "mcp" => validate_mcp_tool_definition(tool)?,
            "function" => validate_function_tool(tool)?,
            _ => {
                return Err(ApiError::unsupported(format!(
                    "tool type '{ty}' is not implemented; use MCP or function tools"
                )));
            }
        }
    }
    Ok(())
}

pub(crate) fn validate_functions_field(body: &Value) -> Result<(), ApiError> {
    let Some(functions) = body.get("functions") else {
        return Ok(());
    };
    let Value::Array(functions) = functions else {
        return Err(ApiError::bad_request("functions must be an array"));
    };
    for function in functions {
        validate_function_definition(function, "functions entries")?;
    }
    Ok(())
}

pub(crate) fn validate_tool_choice_field(body: &Value) -> Result<(), ApiError> {
    let Some(tool_choice) = body.get("tool_choice") else {
        return Ok(());
    };
    match tool_choice {
        Value::Null => Ok(()),
        Value::String(choice) if matches!(choice.as_str(), "none" | "auto" | "required") => Ok(()),
        Value::Object(_) => Ok(()),
        _ => Err(ApiError::bad_request(
            "tool_choice must be none, auto, required, or an object",
        )),
    }
}

pub(crate) fn mcp_tool_invocation_from_request(
    body: &Value,
) -> Result<Option<McpToolInvocation>, ApiError> {
    if let Some(value) = body
        .get("mcp_tool")
        .or_else(|| body.get("mcp_call"))
        .or_else(|| body.get("tool_call"))
    {
        if function_tool_choice_object(value) {
            return Ok(None);
        }
        return parse_mcp_tool_invocation_value(value, body).map(Some);
    }

    if let Some(Value::Object(_)) = body.get("tool_choice") {
        let value = body.get("tool_choice").expect("checked above");
        if function_tool_choice_object(value) || !mcp_tool_choice_object(value) {
            return Ok(None);
        }
        return parse_mcp_tool_invocation_value(value, body).map(Some);
    }

    if body
        .get("tool_choice")
        .and_then(Value::as_str)
        .is_some_and(|choice| choice == "required")
    {
        return required_single_mcp_tool_invocation(body);
    }

    Ok(None)
}

fn parse_mcp_tool_invocation_value(
    value: &Value,
    request_body: &Value,
) -> Result<McpToolInvocation, ApiError> {
    let allowed_tools = mcp_invocation_string_array(value, "allowed_tools")?;
    let name = mcp_invocation_name(value)
        .or_else(|| {
            value
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
        })
        .map(str::to_string)
        .or_else(|| only_string(&allowed_tools))
        .ok_or_else(|| ApiError::bad_request("MCP tool invocation requires name"))?;
    let mut invocation = McpToolInvocation {
        server_id: mcp_invocation_string(value, "server_id"),
        server_label: mcp_invocation_string(value, "server_label")
            .or_else(|| mcp_invocation_string(value, "server")),
        server_url: mcp_invocation_string(value, "server_url")
            .or_else(|| mcp_invocation_string(value, "endpoint"))
            .or_else(|| mcp_invocation_string(value, "url")),
        connector_id: mcp_invocation_string(value, "connector_id"),
        authorization: mcp_invocation_string(value, "authorization"),
        allowed_tools,
        require_approval: mcp_invocation_value(value, "require_approval").unwrap_or(Value::Null),
        name,
        arguments: mcp_invocation_arguments(value)?,
    };
    if invocation.server_id.is_none()
        && invocation.server_label.is_none()
        && invocation.server_url.is_none()
        && invocation.connector_id.is_none()
    {
        fill_mcp_server_from_tools(request_body, &mut invocation)?;
    } else {
        fill_mcp_policy_from_tools(request_body, &mut invocation)?;
    }
    if invocation.require_approval.is_null() {
        invocation.require_approval = json!("always");
    }
    Ok(invocation)
}

fn required_single_mcp_tool_invocation(
    body: &Value,
) -> Result<Option<McpToolInvocation>, ApiError> {
    let tools = body
        .get("tools")
        .and_then(Value::as_array)
        .ok_or_else(|| ApiError::bad_request("tool_choice required needs tools"))?;
    let mut mcp_tools = tools
        .iter()
        .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("mcp"));
    let Some(tool) = mcp_tools.next() else {
        return Ok(None);
    };
    if mcp_tools.next().is_some() {
        return Err(ApiError::unsupported(
            "tool_choice required needs an explicit MCP tool name when multiple MCP tools are provided",
        ));
    }
    parse_mcp_tool_invocation_value(tool, body).map(Some)
}

fn fill_mcp_server_from_tools(
    body: &Value,
    invocation: &mut McpToolInvocation,
) -> Result<(), ApiError> {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return Ok(());
    };
    let mut matches = tools.iter().filter(|tool| {
        tool.get("type").and_then(Value::as_str) == Some("mcp")
            && mcp_invocation_name(tool).is_none_or(|name| name == invocation.name)
    });
    let Some(tool) = matches.next() else {
        return Ok(());
    };
    if matches.next().is_some() {
        return Err(ApiError::unsupported(
            "MCP tool invocation must include server_id, server_label, server_url, or connector_id when multiple matching tools exist",
        ));
    }
    invocation.server_id = invocation
        .server_id
        .clone()
        .or_else(|| mcp_invocation_string(tool, "server_id"));
    invocation.server_label = invocation
        .server_label
        .clone()
        .or_else(|| mcp_invocation_string(tool, "server_label"));
    invocation.server_url = invocation
        .server_url
        .clone()
        .or_else(|| mcp_invocation_string(tool, "server_url"))
        .or_else(|| mcp_invocation_string(tool, "endpoint"))
        .or_else(|| mcp_invocation_string(tool, "url"));
    invocation.connector_id = invocation
        .connector_id
        .clone()
        .or_else(|| mcp_invocation_string(tool, "connector_id"));
    fill_mcp_policy_from_tool(tool, invocation)?;
    Ok(())
}

fn fill_mcp_policy_from_tools(
    body: &Value,
    invocation: &mut McpToolInvocation,
) -> Result<(), ApiError> {
    let Some(tools) = body.get("tools").and_then(Value::as_array) else {
        return Ok(());
    };
    if let Some(tool) = tools.iter().find(|tool| {
        tool.get("type").and_then(Value::as_str) == Some("mcp")
            && (mcp_invocation_string(tool, "server_id") == invocation.server_id
                || mcp_invocation_string(tool, "server_label") == invocation.server_label
                || mcp_invocation_string(tool, "server_url") == invocation.server_url
                || mcp_invocation_string(tool, "connector_id") == invocation.connector_id)
    }) {
        fill_mcp_policy_from_tool(tool, invocation)?;
    }
    Ok(())
}

fn fill_mcp_policy_from_tool(
    tool: &Value,
    invocation: &mut McpToolInvocation,
) -> Result<(), ApiError> {
    if invocation.authorization.is_none() {
        invocation.authorization = mcp_invocation_string(tool, "authorization");
    }
    if invocation.allowed_tools.is_empty() {
        invocation.allowed_tools = mcp_invocation_string_array(tool, "allowed_tools")?;
    }
    if invocation.require_approval.is_null()
        && let Some(require_approval) = mcp_invocation_value(tool, "require_approval")
    {
        invocation.require_approval = require_approval;
    }
    Ok(())
}

fn mcp_invocation_name(value: &Value) -> Option<&str> {
    value
        .get("name")
        .or_else(|| value.get("tool"))
        .or_else(|| value.get("tool_name"))
        .and_then(Value::as_str)
}

fn mcp_invocation_string(value: &Value, name: &'static str) -> Option<String> {
    value
        .get(name)
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            value
                .get("mcp")
                .and_then(|mcp| mcp.get(name))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

fn mcp_invocation_value(value: &Value, name: &'static str) -> Option<Value> {
    value
        .get(name)
        .cloned()
        .or_else(|| value.get("mcp").and_then(|mcp| mcp.get(name)).cloned())
}

fn mcp_invocation_string_array(value: &Value, name: &'static str) -> Result<Vec<String>, ApiError> {
    let Some(value) = mcp_invocation_value(value, name) else {
        return Ok(Vec::new());
    };
    string_vec_from_value(&value, name)
}

fn mcp_invocation_arguments(value: &Value) -> Result<Value, ApiError> {
    let arguments = value
        .get("arguments")
        .or_else(|| value.get("args"))
        .or_else(|| value.get("input"))
        .or_else(|| {
            value
                .get("function")
                .and_then(|function| function.get("arguments"))
        })
        .cloned()
        .unwrap_or_else(|| json!({}));
    match arguments {
        Value::String(text) => serde_json::from_str(&text)
            .map_err(|err| ApiError::bad_request(format!("tool arguments JSON is invalid: {err}"))),
        other => Ok(other),
    }
}

fn mcp_tool_choice_object(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("mcp")
        || value.get("mcp").is_some()
        || mcp_invocation_string(value, "server_id").is_some()
        || mcp_invocation_string(value, "server_label").is_some()
        || mcp_invocation_string(value, "server_url").is_some()
        || mcp_invocation_string(value, "connector_id").is_some()
        || mcp_invocation_string(value, "endpoint").is_some()
        || mcp_invocation_string(value, "url").is_some()
}

fn function_tool_choice_object(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("function") || value.get("function").is_some()
}

fn resolve_mcp_server_for_invocation(
    state: &AppState,
    invocation: &McpToolInvocation,
) -> Result<McpServerRecord, ApiError> {
    if let Some(server_id) = invocation.server_id.as_deref() {
        let servers = lock_mcp_servers(state)?;
        let record = servers.get(server_id).cloned().ok_or_else(|| {
            ApiError::not_found(format!("MCP server '{server_id}' does not exist"))
        })?;
        return Ok(apply_mcp_invocation_overrides(record, invocation));
    }

    if let Some(label) = invocation.server_label.as_deref() {
        let servers = lock_mcp_servers(state)?;
        if let Some(record) = servers.get(label).cloned() {
            return Ok(apply_mcp_invocation_overrides(record, invocation));
        }
        if let Some(record) = servers
            .values()
            .find(|record| record.id == label || record.endpoint == label)
            .cloned()
        {
            return Ok(apply_mcp_invocation_overrides(record, invocation));
        }
        if invocation.server_url.is_none() && invocation.connector_id.is_none() {
            return Err(ApiError::not_found(format!(
                "MCP server label '{label}' does not exist"
            )));
        }
    }

    if let Some(server_url) = invocation.server_url.as_deref() {
        return Ok(apply_mcp_invocation_overrides(
            transient_mcp_server_record(
                server_url,
                infer_mcp_transport(Some(server_url), None, None),
                invocation,
            ),
            invocation,
        ));
    }

    if let Some(connector_id) = invocation.connector_id.as_deref() {
        return Ok(apply_mcp_invocation_overrides(
            transient_mcp_server_record(
                &format!("connector:{connector_id}"),
                "connector".to_string(),
                invocation,
            ),
            invocation,
        ));
    }

    let servers = lock_mcp_servers(state)?;
    if servers.len() == 1 {
        let record = servers.values().next().expect("len checked").clone();
        return Ok(apply_mcp_invocation_overrides(record, invocation));
    }
    Err(ApiError::bad_request(
        "MCP tool invocation requires server_id, server_label, server_url, or connector_id",
    ))
}

fn transient_mcp_server_record(
    endpoint: &str,
    transport: String,
    invocation: &McpToolInvocation,
) -> McpServerRecord {
    McpServerRecord {
        id: invocation
            .server_label
            .clone()
            .unwrap_or_else(|| format!("mcp-transient-{:016x}", stable_hash_str(endpoint))),
        created: unix_seconds(),
        updated: unix_seconds(),
        transport,
        endpoint: endpoint.to_string(),
        message_endpoint: None,
        command: None,
        args: Vec::new(),
        cwd: None,
        env: HashMap::new(),
        authorization: invocation.authorization.clone(),
        protocol_version: MCP_PROTOCOL_VERSION.to_string(),
        session_id: None,
        connector_id: invocation.connector_id.clone(),
        allowed_tools: invocation.allowed_tools.clone(),
        require_approval: invocation.require_approval.clone(),
        status: "transient".to_string(),
        capabilities: Value::Null,
        tools: json!([]),
        last_error: None,
    }
}

fn apply_mcp_invocation_overrides(
    mut record: McpServerRecord,
    invocation: &McpToolInvocation,
) -> McpServerRecord {
    if let Some(server_url) = invocation.server_url.as_ref() {
        record.endpoint = server_url.clone();
        record.transport =
            infer_mcp_transport(Some(server_url), None, invocation.connector_id.as_deref());
    }
    if let Some(authorization) = invocation.authorization.as_ref() {
        record.authorization = Some(authorization.clone());
    }
    if let Some(connector_id) = invocation.connector_id.as_ref() {
        record.connector_id = Some(connector_id.clone());
        if record.endpoint.is_empty() {
            record.endpoint = format!("connector:{connector_id}");
        }
    }
    if !invocation.allowed_tools.is_empty() {
        record.allowed_tools = invocation.allowed_tools.clone();
    }
    if !invocation.require_approval.is_null() {
        record.require_approval = invocation.require_approval.clone();
    }
    record
}

pub(crate) fn augment_prompt_with_mcp_tool(
    mut prompt: String,
    tool_result: Option<&McpToolResult>,
    body: &Value,
) -> String {
    let tool_context = match tool_result {
        Some(result) => format!(
            "MCP tool result from server '{}' tool '{}':\n{}\n",
            result.server_label.as_deref().unwrap_or(&result.server_id),
            result.name,
            result.result
        ),
        None => mcp_tools_prompt_catalog(body).unwrap_or_default(),
    };
    if tool_context.is_empty() {
        return prompt;
    }
    let assistant_marker = "Assistant:";
    if prompt.ends_with(assistant_marker) {
        let new_len = prompt.len().saturating_sub(assistant_marker.len());
        prompt.truncate(new_len);
    }
    prompt.push_str(&tool_context);
    prompt.push_str(assistant_marker);
    prompt
}

fn mcp_tools_prompt_catalog(body: &Value) -> Option<String> {
    let mut lines = Vec::new();
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for tool in tools {
            match tool
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("function")
            {
                "mcp" => {
                    let allowed = mcp_invocation_string_array(tool, "allowed_tools")
                        .unwrap_or_default()
                        .join(", ");
                    let server = mcp_invocation_string(tool, "server_label")
                        .or_else(|| mcp_invocation_string(tool, "server_id"))
                        .or_else(|| mcp_invocation_string(tool, "connector_id"))
                        .or_else(|| mcp_invocation_string(tool, "server_url"))
                        .unwrap_or_else(|| "registered-mcp-server".to_string());
                    let name = mcp_invocation_name(tool)
                        .map(str::to_string)
                        .or_else(|| {
                            only_string(
                                &mcp_invocation_string_array(tool, "allowed_tools")
                                    .unwrap_or_default(),
                            )
                        })
                        .unwrap_or_else(|| "remote-tools".to_string());
                    let mut line = format!("- MCP {name} on {server}");
                    if let Some(description) = mcp_invocation_string(tool, "server_description")
                        .or_else(|| mcp_invocation_string(tool, "description"))
                    {
                        line.push_str(": ");
                        line.push_str(&description);
                    }
                    if !allowed.is_empty() {
                        line.push_str("; allowed tools: ");
                        line.push_str(&allowed);
                    }
                    if let Some(defer_loading) = tool.get("defer_loading").and_then(Value::as_bool)
                        && defer_loading
                    {
                        line.push_str("; defer_loading: true");
                    }
                    lines.push(line);
                }
                "function" => {
                    if let Some(line) =
                        function_definition_line(tool.get("function").unwrap_or(tool), "function")
                    {
                        lines.push(line);
                    }
                }
                _ => {}
            }
        }
    }
    if let Some(functions) = body.get("functions").and_then(Value::as_array) {
        for function in functions {
            if let Some(line) = function_definition_line(function, "function") {
                lines.push(line);
            }
        }
    }
    if lines.is_empty() {
        None
    } else {
        Some(format!(
            "Available tools, if needed:\n{}\n",
            lines.join("\n")
        ))
    }
}

fn validate_mcp_tool_definition(tool: &Value) -> Result<(), ApiError> {
    let Value::Object(_) = tool else {
        return Err(ApiError::bad_request(
            "MCP tool definitions must be objects",
        ));
    };
    let has_registered = mcp_invocation_string(tool, "server_id").is_some()
        || mcp_invocation_string(tool, "server_label").is_some();
    let has_remote = mcp_invocation_string(tool, "server_url")
        .or_else(|| mcp_invocation_string(tool, "endpoint"))
        .or_else(|| mcp_invocation_string(tool, "url"))
        .is_some();
    let has_connector = mcp_invocation_string(tool, "connector_id").is_some();
    if !(has_registered || has_remote || has_connector) {
        return Err(ApiError::bad_request(
            "MCP tools require server_label/server_id, server_url, or connector_id",
        ));
    }
    if tool.get("allowed_tools").is_some() {
        mcp_invocation_string_array(tool, "allowed_tools")?;
    }
    if let Some(require_approval) = mcp_invocation_value(tool, "require_approval") {
        validate_mcp_approval_policy(&require_approval)?;
    }
    if let Some(value) = tool.get("defer_loading")
        && !value.is_boolean()
    {
        return Err(ApiError::bad_request("defer_loading must be a boolean"));
    }
    Ok(())
}

fn validate_mcp_approval_policy(value: &Value) -> Result<(), ApiError> {
    match value {
        Value::String(policy) if matches!(policy.as_str(), "always" | "never") => Ok(()),
        Value::Object(object) => {
            for (key, policy) in object {
                if !matches!(key.as_str(), "always" | "never") {
                    return Err(ApiError::bad_request(
                        "require_approval object keys must be always or never",
                    ));
                }
                let tool_names = policy.get("tool_names").ok_or_else(|| {
                    ApiError::bad_request("require_approval policy requires tool_names")
                })?;
                string_vec_from_value(tool_names, "tool_names")?;
            }
            Ok(())
        }
        _ => Err(ApiError::bad_request(
            "require_approval must be always, never, or a policy object",
        )),
    }
}

fn validate_function_tool(tool: &Value) -> Result<(), ApiError> {
    validate_function_definition(
        tool.get("function").unwrap_or(tool),
        "function tool definitions",
    )
}

fn validate_function_definition(function: &Value, context: &str) -> Result<(), ApiError> {
    let Value::Object(_) = function else {
        return Err(ApiError::bad_request(format!("{context} must be objects")));
    };
    let Some(name) = function.get("name").and_then(Value::as_str) else {
        return Err(ApiError::bad_request(format!(
            "{context} require a function name"
        )));
    };
    if name.trim().is_empty() {
        return Err(ApiError::bad_request(format!(
            "{context} require a non-empty function name"
        )));
    }
    Ok(())
}

fn function_definition_line(function: &Value, label: &str) -> Option<String> {
    let name = function.get("name").and_then(Value::as_str)?;
    let mut line = format!("- {label} {name}");
    if let Some(description) = function.get("description").and_then(Value::as_str) {
        if !description.trim().is_empty() {
            line.push_str(": ");
            line.push_str(description);
        }
    }
    if let Some(parameters) = function.get("parameters") {
        if !parameters.is_null() {
            line.push_str("; parameters: ");
            line.push_str(&parameters.to_string());
        }
    }
    Some(line)
}

pub(crate) fn mcp_tool_result_json(result: &McpToolResult) -> Value {
    json!({
        "server_id": result.server_id,
        "server_label": result.server_label,
        "name": result.name,
        "arguments": result.arguments,
        "result": result.result
    })
}

pub(crate) fn mcp_approval_request_json(request: &McpApprovalRequest) -> Value {
    json!({
        "id": request.id,
        "type": "mcp_approval_request",
        "arguments": request.arguments.to_string(),
        "name": request.name,
        "server_id": request.server_id,
        "server_label": request.server_label
    })
}

fn mcp_approval_request(
    state: &AppState,
    record: &McpServerRecord,
    invocation: &McpToolInvocation,
) -> McpApprovalRequest {
    McpApprovalRequest {
        id: state.next_response_id("mcpr"),
        server_id: record.id.clone(),
        server_label: invocation
            .server_label
            .clone()
            .unwrap_or_else(|| record.id.clone()),
        name: invocation.name.clone(),
        arguments: invocation.arguments.clone(),
    }
}

fn mcp_tool_needs_approval(policy: &Value, tool_name: &str) -> bool {
    match policy {
        Value::String(policy) if policy == "never" => false,
        Value::String(policy) if policy == "always" => true,
        Value::Object(object) => {
            if policy_names_contain(object.get("never"), tool_name) {
                return false;
            }
            if policy_names_contain(object.get("always"), tool_name) {
                return true;
            }
            true
        }
        _ => true,
    }
}

fn policy_names_contain(value: Option<&Value>, tool_name: &str) -> bool {
    value
        .and_then(|value| value.get("tool_names"))
        .and_then(Value::as_array)
        .is_some_and(|names| names.iter().any(|name| name.as_str() == Some(tool_name)))
}

fn request_contains_mcp_approval(body: &Value) -> Result<bool, ApiError> {
    let mut approved = false;
    for item in response_input_like_items(body) {
        if item.get("type").and_then(Value::as_str) != Some("mcp_approval_response") {
            continue;
        }
        match item.get("approve").and_then(Value::as_bool) {
            Some(true) => approved = true,
            Some(false) => {
                return Err(ApiError::bad_request("MCP tool call approval was denied"));
            }
            None => {
                return Err(ApiError::bad_request(
                    "mcp_approval_response requires approve",
                ));
            }
        }
    }
    Ok(approved)
}

fn response_input_like_items(body: &Value) -> Vec<&Value> {
    match body.get("input") {
        Some(Value::Array(items)) => items.iter().collect(),
        Some(item) => vec![item],
        None => Vec::new(),
    }
}

fn validate_mcp_tool_allowed(allowed_tools: &[String], name: &str) -> Result<(), ApiError> {
    if !allowed_tools.is_empty() && !allowed_tools.iter().any(|allowed| allowed == name) {
        return Err(ApiError::bad_request(format!(
            "MCP tool '{name}' is not included in allowed_tools"
        )));
    }
    Ok(())
}

fn call_mcp_tool_sync(
    record: &McpServerRecord,
    name: &str,
    arguments: Value,
) -> Result<Value, ApiError> {
    validate_mcp_tool_allowed(&record.allowed_tools, name)?;
    let mut record = record.clone();
    match record.transport.as_str() {
        "streamable_http" | "http" | "https" | "sse" | "http_sse" => {
            ensure_mcp_http_initialized(&mut record)?;
            let endpoint = record
                .message_endpoint
                .as_deref()
                .unwrap_or(record.endpoint.as_str())
                .to_string();
            mcp_http_json_rpc(
                &record,
                &endpoint,
                "tools/call",
                json!({
                    "name": name,
                    "arguments": arguments
                }),
            )
        }
        "stdio" => mcp_stdio_json_rpc(
            &record,
            "tools/call",
            json!({
                "name": name,
                "arguments": arguments
            }),
        ),
        "connector" => Err(ApiError::unsupported(format!(
            "connector_id '{}' is accepted for API compatibility, but local connector execution requires a server_url-backed MCP server",
            record.connector_id.as_deref().unwrap_or("unknown")
        ))),
        other => Err(ApiError::unsupported(format!(
            "MCP transport '{other}' is not implemented"
        ))),
    }
}

#[derive(Clone, Debug)]
struct McpProbe {
    capabilities: Value,
    tools: Value,
    protocol_version: String,
    session_id: Option<String>,
    message_endpoint: Option<String>,
}

fn probe_mcp_server(record: &McpServerRecord) -> Result<McpProbe, ApiError> {
    match record.transport.as_str() {
        "streamable_http" | "http" | "https" | "sse" | "http_sse" => probe_mcp_http_server(record),
        "stdio" => probe_mcp_stdio_server(record),
        "connector" => Ok(McpProbe {
            capabilities: json!({"tools": {}}),
            tools: json!([]),
            protocol_version: record.protocol_version.clone(),
            session_id: None,
            message_endpoint: None,
        }),
        other => Err(ApiError::unsupported(format!(
            "MCP transport '{other}' is not implemented"
        ))),
    }
}

fn probe_mcp_http_server(record: &McpServerRecord) -> Result<McpProbe, ApiError> {
    let mut record = record.clone();
    let initialized = initialize_mcp_http_session(&mut record)?;
    let capabilities = initialized
        .get("capabilities")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let tools = mcp_http_json_rpc(
        &record,
        record
            .message_endpoint
            .as_deref()
            .unwrap_or(record.endpoint.as_str()),
        "tools/list",
        json!({}),
    )
    .map(|result| result.get("tools").cloned().unwrap_or_else(|| json!([])))
    .unwrap_or_else(|_| json!([]));
    Ok(McpProbe {
        capabilities,
        tools,
        protocol_version: record.protocol_version,
        session_id: record.session_id,
        message_endpoint: record.message_endpoint,
    })
}

fn ensure_mcp_http_initialized(record: &mut McpServerRecord) -> Result<(), ApiError> {
    if record.status == "connected" && !record.capabilities.is_null() {
        return Ok(());
    }
    let initialized = initialize_mcp_http_session(record)?;
    record.capabilities = initialized
        .get("capabilities")
        .cloned()
        .unwrap_or_else(|| json!({}));
    record.status = "connected".to_string();
    Ok(())
}

fn initialize_mcp_http_session(record: &mut McpServerRecord) -> Result<Value, ApiError> {
    let params = json!({
        "protocolVersion": record.protocol_version,
        "capabilities": {},
        "clientInfo": {
            "name": "nerva",
            "version": env!("CARGO_PKG_VERSION")
        }
    });
    let mut endpoint = record.endpoint.clone();
    let initialized =
        match mcp_http_json_rpc_outcome(record, &endpoint, "initialize", params.clone()) {
            Ok(outcome) => {
                apply_http_outcome(record, &outcome);
                outcome.result
            }
            Err(err)
                if err
                    .status
                    .is_some_and(|status| matches!(status, 400 | 404 | 405)) =>
            {
                let legacy_endpoint = discover_legacy_sse_message_endpoint(record)?;
                record.message_endpoint = Some(legacy_endpoint.clone());
                endpoint = legacy_endpoint;
                let outcome = mcp_http_json_rpc_outcome(record, &endpoint, "initialize", params)
                    .map_err(McpHttpCallError::into_api)?;
                apply_http_outcome(record, &outcome);
                outcome.result
            }
            Err(err) => return Err(err.into_api()),
        };
    if let Some(protocol) = initialized.get("protocolVersion").and_then(Value::as_str) {
        record.protocol_version = protocol.to_string();
    }
    let _ = mcp_http_notification(record, &endpoint, "notifications/initialized", json!({}));
    Ok(initialized)
}

fn apply_http_outcome(record: &mut McpServerRecord, outcome: &McpHttpOutcome) {
    if let Some(session_id) = outcome.session_id.as_ref() {
        record.session_id = Some(session_id.clone());
    }
}

fn mcp_http_json_rpc(
    record: &McpServerRecord,
    endpoint: &str,
    method: &str,
    params: Value,
) -> Result<Value, ApiError> {
    mcp_http_json_rpc_outcome(record, endpoint, method, params)
        .map(|outcome| outcome.result)
        .map_err(McpHttpCallError::into_api)
}

fn mcp_http_notification(
    record: &McpServerRecord,
    endpoint: &str,
    method: &str,
    params: Value,
) -> Result<(), ApiError> {
    mcp_http_json_rpc_notification_outcome(record, endpoint, method, params)
        .map(|_| ())
        .map_err(McpHttpCallError::into_api)
}

#[derive(Clone, Debug)]
struct McpHttpOutcome {
    result: Value,
    session_id: Option<String>,
}

#[derive(Debug)]
struct McpHttpCallError {
    status: Option<u16>,
    message: String,
}

impl McpHttpCallError {
    fn into_api(self) -> ApiError {
        match self.status {
            Some(status) => ApiError::bad_gateway(format!(
                "MCP endpoint returned HTTP status {status}: {}",
                self.message
            )),
            None => ApiError::bad_gateway(self.message),
        }
    }
}

fn mcp_http_json_rpc_outcome(
    record: &McpServerRecord,
    endpoint: &str,
    method: &str,
    params: Value,
) -> Result<McpHttpOutcome, McpHttpCallError> {
    let id = format!("nerva-{:x}-{:x}", unix_seconds(), stable_hash_str(method));
    let body = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params
    });
    mcp_http_send_json_rpc(record, endpoint, body, true)
}

fn mcp_http_json_rpc_notification_outcome(
    record: &McpServerRecord,
    endpoint: &str,
    method: &str,
    params: Value,
) -> Result<McpHttpOutcome, McpHttpCallError> {
    let body = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params
    });
    mcp_http_send_json_rpc(record, endpoint, body, false)
}

fn mcp_http_send_json_rpc(
    record: &McpServerRecord,
    endpoint: &str,
    body: Value,
    expects_response: bool,
) -> Result<McpHttpOutcome, McpHttpCallError> {
    let body = body.to_string();
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(DEFAULT_MCP_CONNECT_TIMEOUT_SECS))
        .timeout_read(Duration::from_secs(DEFAULT_MCP_READ_TIMEOUT_SECS))
        .timeout_write(Duration::from_secs(DEFAULT_MCP_CONNECT_TIMEOUT_SECS))
        .build();
    let mut request = agent
        .post(endpoint)
        .set("Content-Type", "application/json")
        .set("Accept", "application/json, text/event-stream")
        .set("User-Agent", concat!("nerva/", env!("CARGO_PKG_VERSION")))
        .set("MCP-Protocol-Version", &record.protocol_version);
    if let Some(session_id) = record.session_id.as_deref() {
        request = request.set("Mcp-Session-Id", session_id);
    }
    if let Some(authorization) = record.authorization.as_deref() {
        request = request.set("Authorization", &authorization_header(authorization));
    }
    let response = match request.send_string(&body) {
        Ok(response) => response,
        Err(ureq::Error::Status(status, response)) => {
            let message = response.into_string().unwrap_or_default();
            return Err(McpHttpCallError {
                status: Some(status),
                message,
            });
        }
        Err(err) => {
            return Err(McpHttpCallError {
                status: None,
                message: format!("failed to call MCP endpoint: {err}"),
            });
        }
    };
    let status = response.status();
    let content_type = response.header("content-type").unwrap_or("").to_string();
    let session_id = response.header("Mcp-Session-Id").map(str::to_string);
    if !expects_response && status == 202 {
        return Ok(McpHttpOutcome {
            result: Value::Null,
            session_id,
        });
    }
    let text = response.into_string().map_err(|err| McpHttpCallError {
        status: None,
        message: format!("failed to read MCP response: {err}"),
    })?;
    let result = parse_mcp_response_text(&content_type, &text).map_err(|err| McpHttpCallError {
        status: None,
        message: err.message,
    })?;
    Ok(McpHttpOutcome { result, session_id })
}

fn discover_legacy_sse_message_endpoint(record: &McpServerRecord) -> Result<String, ApiError> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(DEFAULT_MCP_CONNECT_TIMEOUT_SECS))
        .timeout_read(Duration::from_secs(DEFAULT_MCP_READ_TIMEOUT_SECS))
        .build();
    let mut request = agent
        .get(&record.endpoint)
        .set("Accept", "text/event-stream")
        .set("User-Agent", concat!("nerva/", env!("CARGO_PKG_VERSION")));
    if let Some(authorization) = record.authorization.as_deref() {
        request = request.set("Authorization", &authorization_header(authorization));
    }
    let response = request.call().map_err(|err| {
        ApiError::bad_gateway(format!("failed to discover legacy MCP SSE endpoint: {err}"))
    })?;
    if response.status() != 200 {
        return Err(ApiError::bad_gateway(format!(
            "legacy MCP SSE endpoint returned HTTP status {}",
            response.status()
        )));
    }
    let mut reader = BufReader::new(response.into_reader());
    let mut event = String::new();
    let mut data = String::new();
    for _ in 0..256 {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line).map_err(|err| {
            ApiError::bad_gateway(format!("failed to read MCP SSE endpoint: {err}"))
        })?;
        if bytes == 0 {
            break;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            if event == "endpoint" && !data.trim().is_empty() {
                return Ok(absolutize_endpoint(&record.endpoint, data.trim()));
            }
            event.clear();
            data.clear();
            continue;
        }
        if let Some(value) = trimmed.strip_prefix("event:") {
            event = value.trim().to_string();
        } else if let Some(value) = trimmed.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(value.trim());
        }
    }
    Err(ApiError::bad_gateway(
        "legacy MCP SSE stream did not provide an endpoint event",
    ))
}

fn probe_mcp_stdio_server(record: &McpServerRecord) -> Result<McpProbe, ApiError> {
    let initialized = mcp_stdio_json_rpc(record, "initialize", initialize_params(record))?;
    let capabilities = initialized
        .get("capabilities")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let protocol_version = initialized
        .get("protocolVersion")
        .and_then(Value::as_str)
        .unwrap_or(record.protocol_version.as_str())
        .to_string();
    let tools = mcp_stdio_json_rpc(record, "tools/list", json!({}))
        .map(|result| result.get("tools").cloned().unwrap_or_else(|| json!([])))
        .unwrap_or_else(|_| json!([]));
    Ok(McpProbe {
        capabilities,
        tools,
        protocol_version,
        session_id: None,
        message_endpoint: None,
    })
}

fn initialize_params(record: &McpServerRecord) -> Value {
    json!({
        "protocolVersion": record.protocol_version,
        "capabilities": {},
        "clientInfo": {
            "name": "nerva",
            "version": env!("CARGO_PKG_VERSION")
        }
    })
}

fn mcp_stdio_json_rpc(
    record: &McpServerRecord,
    method: &str,
    params: Value,
) -> Result<Value, ApiError> {
    let command = record
        .command
        .as_deref()
        .ok_or_else(|| ApiError::bad_request("stdio MCP transport requires command"))?;
    let mut cmd = Command::new(command);
    cmd.args(&record.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    if let Some(cwd) = record.cwd.as_deref() {
        cmd.current_dir(cwd);
    }
    for (key, value) in &record.env {
        cmd.env(key, value);
    }
    let mut child = cmd
        .spawn()
        .map_err(|err| ApiError::bad_gateway(format!("failed to start MCP stdio server: {err}")))?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| ApiError::bad_gateway("MCP stdio server stdin unavailable"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| ApiError::bad_gateway("MCP stdio server stdout unavailable"))?;
    let mut reader = BufReader::new(stdout);
    let init_id = "nerva-init";
    write_stdio_json_rpc(&mut stdin, init_id, "initialize", initialize_params(record))?;
    let initialized = read_stdio_json_rpc_response(&mut reader, init_id)?;
    write_stdio_notification(&mut stdin, "notifications/initialized", json!({}))?;
    let result = if method == "initialize" {
        initialized
    } else {
        let id = format!("nerva-{}", stable_hash_str(method));
        write_stdio_json_rpc(&mut stdin, &id, method, params)?;
        read_stdio_json_rpc_response(&mut reader, &id)?
    };
    drop(stdin);
    let _ = child.kill();
    let _ = child.wait();
    Ok(result)
}

fn write_stdio_json_rpc(
    stdin: &mut impl Write,
    id: &str,
    method: &str,
    params: Value,
) -> Result<(), ApiError> {
    let message = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params
    });
    writeln!(stdin, "{message}")
        .map_err(|err| ApiError::bad_gateway(format!("failed to write MCP stdio request: {err}")))
}

fn write_stdio_notification(
    stdin: &mut impl Write,
    method: &str,
    params: Value,
) -> Result<(), ApiError> {
    let message = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params
    });
    writeln!(stdin, "{message}").map_err(|err| {
        ApiError::bad_gateway(format!("failed to write MCP stdio notification: {err}"))
    })
}

fn read_stdio_json_rpc_response(
    reader: &mut impl BufRead,
    expected_id: &str,
) -> Result<Value, ApiError> {
    for _ in 0..1024 {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line).map_err(|err| {
            ApiError::bad_gateway(format!("failed to read MCP stdio response: {err}"))
        })?;
        if bytes == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line.trim()).map_err(|err| {
            ApiError::bad_gateway(format!("invalid MCP stdio JSON-RPC response: {err}"))
        })?;
        if value.get("id").and_then(Value::as_str) != Some(expected_id) {
            continue;
        }
        return json_rpc_result(value);
    }
    Err(ApiError::bad_gateway(
        "MCP stdio server closed before returning the requested response",
    ))
}

fn parse_mcp_response_text(content_type: &str, text: &str) -> Result<Value, ApiError> {
    let json_text = if content_type
        .to_ascii_lowercase()
        .starts_with("text/event-stream")
    {
        first_sse_json_rpc_payload(text)
            .ok_or_else(|| ApiError::bad_gateway("MCP SSE response contained no JSON-RPC result"))?
    } else {
        text.trim().to_string()
    };
    let value: Value = serde_json::from_str(&json_text)
        .map_err(|err| ApiError::bad_gateway(format!("invalid MCP JSON-RPC response: {err}")))?;
    json_rpc_result(value)
}

fn json_rpc_result(value: Value) -> Result<Value, ApiError> {
    if let Some(error) = value.get("error") {
        return Err(ApiError::bad_gateway(format!(
            "MCP JSON-RPC error: {}",
            error
        )));
    }
    value
        .get("result")
        .cloned()
        .ok_or_else(|| ApiError::bad_gateway("MCP JSON-RPC response is missing result"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg(test)]
pub(crate) struct HttpEndpoint {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) path: String,
}

#[cfg(test)]
pub(crate) fn parse_http_endpoint(endpoint: &str) -> Result<HttpEndpoint, ApiError> {
    let (default_port, rest) = if let Some(rest) = endpoint.strip_prefix("http://") {
        (80, rest)
    } else if let Some(rest) = endpoint.strip_prefix("https://") {
        (443, rest)
    } else {
        return Err(ApiError::unsupported(
            "MCP HTTP endpoints must start with http:// or https://",
        ));
    };
    let (authority, path) = rest
        .split_once('/')
        .map(|(authority, path)| (authority, format!("/{path}")))
        .unwrap_or((rest, "/".to_string()));
    if authority.is_empty() {
        return Err(ApiError::bad_request("MCP endpoint host must not be empty"));
    }
    let (host, port) = if let Some((host, port)) = authority.rsplit_once(':') {
        let port = port
            .parse::<u16>()
            .map_err(|_| ApiError::bad_request("MCP endpoint port is invalid"))?;
        (host.to_string(), port)
    } else {
        (authority.to_string(), default_port)
    };
    Ok(HttpEndpoint { host, port, path })
}

#[cfg(test)]
pub(crate) fn parse_mcp_http_response(response: &[u8]) -> Result<Value, ApiError> {
    let split = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| ApiError::bad_gateway("MCP response is missing HTTP headers"))?;
    let header_bytes = &response[..split];
    let body_bytes = &response[split + 4..];
    let headers = String::from_utf8_lossy(header_bytes);
    let status_line = headers.lines().next().unwrap_or("");
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|status| status.parse::<u16>().ok())
        .unwrap_or(0);
    if !(200..300).contains(&status) {
        return Err(ApiError::bad_gateway(format!(
            "MCP endpoint returned HTTP status {status}"
        )));
    }
    let chunked = headers
        .lines()
        .any(|line| line.eq_ignore_ascii_case("transfer-encoding: chunked"));
    let body = if chunked {
        decode_chunked_body(body_bytes)?
    } else {
        body_bytes.to_vec()
    };
    let text = String::from_utf8_lossy(&body);
    let content_type = headers
        .lines()
        .find_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .map(|(_, value)| value.trim())
        .unwrap_or("application/json");
    parse_mcp_response_text(content_type, &text)
}

#[cfg(test)]
pub(crate) fn decode_chunked_body(body: &[u8]) -> Result<Vec<u8>, ApiError> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    while offset < body.len() {
        let line_end = body[offset..]
            .windows(2)
            .position(|window| window == b"\r\n")
            .map(|index| offset + index)
            .ok_or_else(|| ApiError::bad_gateway("invalid chunked MCP response"))?;
        let line = String::from_utf8_lossy(&body[offset..line_end]);
        let size_hex = line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|_| ApiError::bad_gateway("invalid chunk size in MCP response"))?;
        offset = line_end + 2;
        if size == 0 {
            break;
        }
        let end = offset
            .checked_add(size)
            .ok_or_else(|| ApiError::bad_gateway("chunk size overflow in MCP response"))?;
        if end > body.len() {
            return Err(ApiError::bad_gateway("truncated chunked MCP response"));
        }
        out.extend_from_slice(&body[offset..end]);
        offset = end.saturating_add(2);
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) fn first_sse_json_payload(text: &str) -> Option<String> {
    sse_events(text).into_iter().find_map(|event| {
        (!event.data.trim().is_empty() && event.data.trim() != "[DONE]")
            .then(|| event.data.trim().to_string())
    })
}

fn first_sse_json_rpc_payload(text: &str) -> Option<String> {
    sse_events(text).into_iter().find_map(|event| {
        let data = event.data.trim();
        if data.is_empty() || data == "[DONE]" {
            return None;
        }
        let Ok(value) = serde_json::from_str::<Value>(data) else {
            return None;
        };
        (value.get("result").is_some() || value.get("error").is_some()).then(|| data.to_string())
    })
}

#[derive(Clone, Debug, Default)]
struct SseEvent {
    event: Option<String>,
    data: String,
    id: Option<String>,
}

fn sse_events(text: &str) -> Vec<SseEvent> {
    let mut events = Vec::new();
    let mut current = SseEvent::default();
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            if current.event.is_some() || !current.data.is_empty() || current.id.is_some() {
                events.push(current);
                current = SseEvent::default();
            }
            continue;
        }
        if let Some(value) = line.strip_prefix("event:") {
            current.event = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("data:") {
            if !current.data.is_empty() {
                current.data.push('\n');
            }
            current.data.push_str(value.trim());
        } else if let Some(value) = line.strip_prefix("id:") {
            current.id = Some(value.trim().to_string());
        }
    }
    if current.event.is_some() || !current.data.is_empty() || current.id.is_some() {
        events.push(current);
    }
    events
}

fn filter_mcp_tools(tools: Value, allowed_tools: &[String]) -> Value {
    if allowed_tools.is_empty() {
        return tools;
    }
    let allowed = allowed_tools
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    match tools {
        Value::Array(tools) => Value::Array(
            tools
                .into_iter()
                .filter(|tool| {
                    tool.get("name")
                        .and_then(Value::as_str)
                        .is_some_and(|name| allowed.contains(name))
                })
                .collect(),
        ),
        other => other,
    }
}

fn infer_mcp_transport(
    endpoint: Option<&str>,
    command: Option<&str>,
    connector_id: Option<&str>,
) -> String {
    if command.is_some() {
        return "stdio".to_string();
    }
    if connector_id.is_some() {
        return "connector".to_string();
    }
    if endpoint.is_some_and(|endpoint| endpoint.contains("/sse")) {
        return "sse".to_string();
    }
    "streamable_http".to_string()
}

fn authorization_header(value: &str) -> String {
    let lower = value.to_ascii_lowercase();
    if lower.starts_with("bearer ") || lower.starts_with("basic ") {
        value.to_string()
    } else {
        format!("Bearer {value}")
    }
}

fn absolutize_endpoint(base: &str, endpoint: &str) -> String {
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        return endpoint.to_string();
    }
    if endpoint.starts_with('/') {
        if let Some(scheme_end) = base.find("://") {
            let authority_start = scheme_end + 3;
            let authority_end = base[authority_start..]
                .find('/')
                .map(|index| authority_start + index)
                .unwrap_or(base.len());
            return format!("{}{}", &base[..authority_end], endpoint);
        }
    }
    let base_dir = base
        .rsplit_once('/')
        .map(|(prefix, _)| prefix)
        .unwrap_or(base);
    format!("{base_dir}/{endpoint}")
}

fn stable_hash_str(value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn optional_string(value: &Value, name: &'static str) -> Result<Option<String>, ApiError> {
    match value.get(name) {
        Some(Value::String(text)) if !text.trim().is_empty() => Ok(Some(text.to_string())),
        Some(Value::String(_)) => Err(ApiError::bad_request(format!("{name} must not be empty"))),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(ApiError::bad_request(format!("{name} must be a string"))),
    }
}

fn string_array(value: &Value, name: &'static str) -> Result<Vec<String>, ApiError> {
    match value.get(name) {
        Some(value) => string_vec_from_value(value, name),
        None => Ok(Vec::new()),
    }
}

fn string_vec_from_value(value: &Value, name: &'static str) -> Result<Vec<String>, ApiError> {
    let Value::Array(values) = value else {
        return Err(ApiError::bad_request(format!("{name} must be an array")));
    };
    values
        .iter()
        .map(|value| match value.as_str() {
            Some(text) if !text.trim().is_empty() => Ok(text.to_string()),
            Some(_) => Err(ApiError::bad_request(format!(
                "{name} entries must not be empty"
            ))),
            None => Err(ApiError::bad_request(format!(
                "{name} entries must be strings"
            ))),
        })
        .collect()
}

fn string_map(value: &Value, name: &'static str) -> Result<HashMap<String, String>, ApiError> {
    let Some(value) = value.get(name) else {
        return Ok(HashMap::new());
    };
    let Value::Object(map) = value else {
        return Err(ApiError::bad_request(format!("{name} must be an object")));
    };
    map.iter()
        .map(|(key, value)| match value.as_str() {
            Some(text) => Ok((key.clone(), text.to_string())),
            None => Err(ApiError::bad_request(format!(
                "{name} values must be strings"
            ))),
        })
        .collect()
}

fn only_string(values: &[String]) -> Option<String> {
    (values.len() == 1).then(|| values[0].clone())
}

fn lock_mcp_servers(
    state: &AppState,
) -> Result<std::sync::MutexGuard<'_, HashMap<String, McpServerRecord>>, ApiError> {
    state
        .mcp_servers
        .lock()
        .map_err(|_| ApiError::internal("MCP server registry lock poisoned"))
}

fn mcp_server_json(record: &McpServerRecord) -> Value {
    json!({
        "id": record.id,
        "object": "mcp_server",
        "created": record.created,
        "updated": record.updated,
        "transport": record.transport,
        "endpoint": record.endpoint,
        "message_endpoint": record.message_endpoint,
        "command": record.command,
        "args": record.args,
        "cwd": record.cwd,
        "env": record.env.keys().collect::<Vec<_>>(),
        "authorization": record.authorization.as_ref().map(|_| "redacted"),
        "protocol_version": record.protocol_version,
        "session_id": record.session_id,
        "connector_id": record.connector_id,
        "allowed_tools": record.allowed_tools,
        "require_approval": record.require_approval,
        "status": record.status,
        "capabilities": record.capabilities,
        "tools": record.tools,
        "last_error": record.last_error
    })
}
