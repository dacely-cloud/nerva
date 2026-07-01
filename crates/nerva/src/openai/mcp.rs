use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, McpServerRecord, McpToolInvocation, McpToolResult, authorize, unix_seconds,
};

pub(crate) async fn register_mcp_server(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        let id = body
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| state.next_response_id("mcp"));
        let transport = body
            .get("transport")
            .and_then(Value::as_str)
            .unwrap_or("streamable_http")
            .to_string();
        let endpoint = body
            .get("endpoint")
            .or_else(|| body.get("url"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ApiError::bad_request("MCP server registration requires endpoint or url")
            })?
            .to_string();
        let probe = body.get("probe").and_then(Value::as_bool).unwrap_or(true);
        let now = unix_seconds();
        let mut record = McpServerRecord {
            id: id.clone(),
            created: now,
            updated: now,
            transport,
            endpoint,
            status: "registered".to_string(),
            capabilities: Value::Null,
            tools: json!([]),
            last_error: None,
        };
        if probe {
            let probe = probe_mcp_server(&record)?;
            record.status = "connected".to_string();
            record.capabilities = probe.capabilities;
            record.tools = probe.tools;
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
        let record = {
            let servers = lock_mcp_servers(&state)?;
            servers
                .get(&id)
                .cloned()
                .ok_or_else(|| ApiError::not_found(format!("MCP server '{id}' does not exist")))?
        };
        let result = call_mcp_tool_sync(&record, name, arguments)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "server_id": id,
            "tool": name,
            "result": result
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn execute_request_mcp_tool(
    state: web::Data<AppState>,
    body: &Value,
) -> Result<Option<McpToolResult>, ApiError> {
    let Some(invocation) = mcp_tool_invocation_from_request(body)? else {
        return Ok(None);
    };
    let record = resolve_mcp_server_for_invocation(&state, &invocation)?;
    let server_id = record.id.clone();
    let name = invocation.name.clone();
    let arguments = invocation.arguments.clone();
    let result_name = name.clone();
    let result_arguments = arguments.clone();
    let result = web::block(move || call_mcp_tool_sync(&record, &name, arguments))
        .await
        .map_err(|err| ApiError::internal(format!("MCP tool task failed: {err}")))??;
    Ok(Some(McpToolResult {
        server_id,
        name: result_name,
        arguments: result_arguments,
        result,
    }))
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
            "mcp" => {}
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
    let name = mcp_invocation_name(value)
        .or_else(|| {
            value
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
        })
        .ok_or_else(|| ApiError::bad_request("MCP tool invocation requires name"))?
        .to_string();
    let mut invocation = McpToolInvocation {
        server_id: mcp_invocation_string(value, "server_id"),
        server_label: mcp_invocation_string(value, "server_label")
            .or_else(|| mcp_invocation_string(value, "server")),
        server_url: mcp_invocation_string(value, "server_url")
            .or_else(|| mcp_invocation_string(value, "endpoint"))
            .or_else(|| mcp_invocation_string(value, "url")),
        name,
        arguments: mcp_invocation_arguments(value)?,
    };
    if invocation.server_id.is_none()
        && invocation.server_label.is_none()
        && invocation.server_url.is_none()
    {
        fill_mcp_server_from_tools(request_body, &mut invocation)?;
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
            "MCP tool invocation must include server_id, server_label, or server_url when multiple matching tools exist",
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
        return servers.get(server_id).cloned().ok_or_else(|| {
            ApiError::not_found(format!("MCP server '{server_id}' does not exist"))
        });
    }

    if let Some(label) = invocation.server_label.as_deref() {
        let servers = lock_mcp_servers(state)?;
        if let Some(record) = servers.get(label).cloned() {
            return Ok(record);
        }
        if let Some(record) = servers
            .values()
            .find(|record| record.id == label || record.endpoint == label)
            .cloned()
        {
            return Ok(record);
        }
        return Err(ApiError::not_found(format!(
            "MCP server label '{label}' does not exist"
        )));
    }

    if let Some(server_url) = invocation.server_url.as_deref() {
        return Ok(McpServerRecord {
            id: format!("mcp-transient-{:016x}", stable_hash_str(server_url)),
            created: unix_seconds(),
            updated: unix_seconds(),
            transport: "streamable_http".to_string(),
            endpoint: server_url.to_string(),
            status: "transient".to_string(),
            capabilities: Value::Null,
            tools: json!([]),
            last_error: None,
        });
    }

    let servers = lock_mcp_servers(state)?;
    if servers.len() == 1 {
        return Ok(servers.values().next().expect("len checked").clone());
    }
    Err(ApiError::bad_request(
        "MCP tool invocation requires server_id, server_label, or server_url",
    ))
}

pub(crate) fn augment_prompt_with_mcp_tool(
    mut prompt: String,
    tool_result: Option<&McpToolResult>,
    body: &Value,
) -> String {
    let tool_context = match tool_result {
        Some(result) => format!(
            "MCP tool result from server '{}' tool '{}':\n{}\n",
            result.server_id, result.name, result.result
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
                    let name = mcp_invocation_name(tool).unwrap_or("unknown");
                    let server = mcp_invocation_string(tool, "server_id")
                        .or_else(|| mcp_invocation_string(tool, "server_label"))
                        .or_else(|| mcp_invocation_string(tool, "server_url"))
                        .unwrap_or_else(|| "registered-mcp-server".to_string());
                    lines.push(format!("- MCP {name} on {server}"));
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
        "name": result.name,
        "arguments": result.arguments,
        "result": result.result
    })
}

fn stable_hash_str(value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

#[derive(Clone, Debug)]
struct McpProbe {
    capabilities: Value,
    tools: Value,
}

fn probe_mcp_server(record: &McpServerRecord) -> Result<McpProbe, ApiError> {
    if !matches!(record.transport.as_str(), "streamable_http" | "http") {
        return Err(ApiError::unsupported(format!(
            "MCP transport '{}' is not implemented; use streamable_http",
            record.transport
        )));
    }
    let initialized = mcp_http_json_rpc(
        &record.endpoint,
        "initialize",
        json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {
                "name": "nerva",
                "version": env!("CARGO_PKG_VERSION")
            }
        }),
    )?;
    let capabilities = initialized
        .get("capabilities")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let tools = mcp_http_json_rpc(&record.endpoint, "tools/list", json!({}))
        .map(|result| result.get("tools").cloned().unwrap_or_else(|| json!([])))
        .unwrap_or_else(|_| json!([]));
    Ok(McpProbe {
        capabilities,
        tools,
    })
}

fn call_mcp_tool_sync(
    record: &McpServerRecord,
    name: &str,
    arguments: Value,
) -> Result<Value, ApiError> {
    if !matches!(record.transport.as_str(), "streamable_http" | "http") {
        return Err(ApiError::unsupported(format!(
            "MCP transport '{}' is not implemented; use streamable_http",
            record.transport
        )));
    }
    mcp_http_json_rpc(
        &record.endpoint,
        "tools/call",
        json!({
            "name": name,
            "arguments": arguments
        }),
    )
}

fn mcp_http_json_rpc(endpoint: &str, method: &str, params: Value) -> Result<Value, ApiError> {
    let target = parse_http_endpoint(endpoint)?;
    let body = json!({
        "jsonrpc": "2.0",
        "id": format!("nerva-{:x}", unix_seconds()),
        "method": method,
        "params": params
    })
    .to_string();
    let address = format!("{}:{}", target.host, target.port);
    let socket_addr = address
        .to_socket_addrs()
        .map_err(|err| ApiError::bad_gateway(format!("failed to resolve MCP endpoint: {err}")))?
        .next()
        .ok_or_else(|| ApiError::bad_gateway("MCP endpoint resolved no addresses"))?;
    let mut stream =
        TcpStream::connect_timeout(&socket_addr, Duration::from_secs(10)).map_err(|err| {
            ApiError::bad_gateway(format!("failed to connect to MCP endpoint: {err}"))
        })?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
    let request = format!(
        concat!(
            "POST {} HTTP/1.1\r\n",
            "Host: {}\r\n",
            "Content-Type: application/json\r\n",
            "Accept: application/json, text/event-stream\r\n",
            "Content-Length: {}\r\n",
            "Connection: close\r\n",
            "\r\n",
            "{}"
        ),
        target.path,
        target.host,
        body.len(),
        body
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|err| ApiError::bad_gateway(format!("failed to write MCP request: {err}")))?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|err| ApiError::bad_gateway(format!("failed to read MCP response: {err}")))?;
    parse_mcp_http_response(&response)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HttpEndpoint {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) path: String,
}

pub(crate) fn parse_http_endpoint(endpoint: &str) -> Result<HttpEndpoint, ApiError> {
    let rest = endpoint.strip_prefix("http://").ok_or_else(|| {
        ApiError::unsupported("MCP streamable_http currently supports http:// endpoints")
    })?;
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
        (authority.to_string(), 80)
    };
    Ok(HttpEndpoint { host, port, path })
}

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
    let json_text = if headers.lines().any(|line| {
        line.to_ascii_lowercase()
            .starts_with("content-type: text/event-stream")
    }) {
        first_sse_json_payload(&text)
            .ok_or_else(|| ApiError::bad_gateway("MCP SSE response contained no JSON data"))?
    } else {
        text.trim().to_string()
    };
    let value: Value = serde_json::from_str(&json_text)
        .map_err(|err| ApiError::bad_gateway(format!("invalid MCP JSON-RPC response: {err}")))?;
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

pub(crate) fn first_sse_json_payload(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let data = line.strip_prefix("data:")?.trim();
        (!data.is_empty() && data != "[DONE]").then(|| data.to_string())
    })
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
        "status": record.status,
        "capabilities": record.capabilities,
        "tools": record.tools,
        "last_error": record.last_error
    })
}
