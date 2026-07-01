use serde_json::json;

use super::{
    GeneratedText, McpToolResult, PromptInput, ReasoningMode, StreamKind, StreamMeta,
    apply_stop_strings, augment_prompt_with_mcp_tool, chat_messages_to_prompt, completion_prompts,
    completion_text, decode_chunked_body, finish_reason, first_sse_json_payload,
    generated_metadata, hash_tokens, mcp_tool_invocation_from_request, normalize_batch_endpoint,
    parse_http_endpoint, parse_mcp_http_response, parse_multipart_file_upload,
    percent_decode_query, prompt_format_for_reasoning, reject_unsupported_generation_options,
    reject_unsupported_generation_options_with_tools, request_n, request_optional_string,
    request_stop_strings, responses_input_to_prompt, send_stream_reasoning_delta, session_json,
    shared_fork_batch_supported, split_generated_reasoning, text_delta,
};
use crate::openai::SessionRecord;
use nerva_model::causal_lm::types::HfCausalLmStopReason;
use nerva_model::hf::tokenizer::PromptFormat;

#[test]
fn parses_string_and_array_completion_prompts() {
    let prompts = completion_prompts(&json!({"prompt": "hello"})).unwrap();
    assert!(matches!(prompts[0], PromptInput::Text { .. }));

    let prompts = completion_prompts(&json!({"prompt": ["a", "b"]})).unwrap();
    assert_eq!(prompts.len(), 2);
    assert!(matches!(prompts[0], PromptInput::Text { .. }));

    let prompts = completion_prompts(&json!({"prompt": [1, 2, 3]})).unwrap();
    assert!(matches!(prompts[0], PromptInput::TokenIds(_)));

    let prompts = completion_prompts(&json!({"prompt": [[1, 2], [3, 4]]})).unwrap();
    assert_eq!(prompts.len(), 2);
    assert!(matches!(prompts[1], PromptInput::TokenIds(_)));
}

#[test]
fn renders_chat_messages_to_text_prompt() {
    let prompt = chat_messages_to_prompt(&json!({
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ]
    }))
    .unwrap();
    assert!(prompt.contains("System: be terse"));
    assert!(prompt.contains("User: hello"));
    assert!(prompt.ends_with("Assistant:"));
}

#[test]
fn parses_responses_input_messages() {
    let prompt = responses_input_to_prompt(&json!({
        "instructions": "be useful",
        "input": [{"role": "user", "content": "hello"}]
    }))
    .unwrap();
    assert!(prompt.contains("System: be useful"));
    assert!(prompt.contains("User: hello"));
}

#[test]
fn parses_stop_strings() {
    assert_eq!(
        request_stop_strings(&json!({"stop": ["END", "STOP"]})).unwrap(),
        vec!["END".to_string(), "STOP".to_string()]
    );
    let (text, stopped) = apply_stop_strings("hello END world".to_string(), &["END".into()]);
    assert_eq!(text, "hello ");
    assert!(stopped);
}

#[test]
fn formats_completion_echo_and_suffix_text() {
    assert_eq!(completion_text("answer", None, None), "answer");
    assert_eq!(
        completion_text("answer", Some("prompt "), Some(" done")),
        "prompt answer done"
    );
}

#[test]
fn serializes_generated_cache_and_session_metadata() {
    let generated = GeneratedText {
        text: "ok".to_string(),
        token_ids: vec![1, 2],
        prompt_tokens: 3,
        finish_reason: "stop",
        prompt_hash: 0x12ab,
        cache_key: "prompt:12ab".to_string(),
        cache_hit: true,
        session_id: Some("sess-1".to_string()),
    };
    assert_eq!(
        generated_metadata(&generated),
        json!({
            "cache_key": "prompt:12ab",
            "cache_hit": true,
            "session_id": "sess-1",
            "prompt_hash": "00000000000012ab"
        })
    );
}

#[test]
fn computes_stream_text_delta() {
    assert_eq!(text_delta("hello", "hello world"), " world");
    assert_eq!(text_delta("abc", "xyz"), "xyz");
}

#[test]
fn splits_deepseek_reasoning_markers() {
    let split = split_generated_reasoning("work</think>answer", ReasoningMode::DeepSeekThinking);
    assert_eq!(split.reasoning, "work");
    assert_eq!(split.content, "answer");

    let split = split_generated_reasoning("<think>why</think>done", ReasoningMode::DeepSeekChat);
    assert_eq!(split.reasoning, "why");
    assert_eq!(split.content, "done");

    let split = split_generated_reasoning("plain", ReasoningMode::DeepSeekChat);
    assert_eq!(split.reasoning, "");
    assert_eq!(split.content, "plain");
}

#[test]
fn maps_deepseek_reasoning_mode_to_prompt_format() {
    assert_eq!(
        prompt_format_for_reasoning(ReasoningMode::None),
        PromptFormat::Auto
    );
    assert_eq!(
        prompt_format_for_reasoning(ReasoningMode::DeepSeekChat),
        PromptFormat::DeepSeekChat
    );
    assert_eq!(
        prompt_format_for_reasoning(ReasoningMode::DeepSeekThinking),
        PromptFormat::DeepSeekThinking
    );
}

#[test]
fn streams_chat_reasoning_delta_as_reasoning_field() {
    let (tx, mut rx) = tokio::sync::mpsc::channel(2);
    let meta = StreamMeta {
        id: "chatcmpl-test".to_string(),
        created: 7,
        model: "deepseek-test".to_string(),
    };
    let mut response_reasoning_started = false;

    assert!(send_stream_reasoning_delta(
        &tx,
        StreamKind::ChatCompletion,
        &meta,
        "thinking",
        &mut response_reasoning_started,
    ));
    drop(tx);

    let frame = rx.blocking_recv().unwrap();
    let frame = std::str::from_utf8(frame.as_ref()).unwrap();
    assert!(frame.contains("\"reasoning\":\"thinking\""));
    assert!(frame.contains("\"reasoning_content\":\"thinking\""));
    assert!(!frame.contains("\"content\":\"thinking\""));
}

#[test]
fn maps_finish_reason_to_openai_values() {
    assert_eq!(finish_reason(HfCausalLmStopReason::EosToken, false), "stop");
    assert_eq!(finish_reason(HfCausalLmStopReason::MaxSteps, true), "stop");
    assert_eq!(
        finish_reason(HfCausalLmStopReason::MaxSteps, false),
        "length"
    );
}

#[test]
fn rejects_unsupported_generation_options() {
    assert!(reject_unsupported_generation_options(&json!({})).is_ok());
    assert!(reject_unsupported_generation_options(&json!({"echo": true})).is_ok());
    assert!(reject_unsupported_generation_options(&json!({"suffix": " done"})).is_ok());
    assert!(reject_unsupported_generation_options(&json!({"presence_penalty": 0.0})).is_ok());
    assert!(reject_unsupported_generation_options(&json!({"presence_penalty": 1.0})).is_err());
    assert!(reject_unsupported_generation_options(&json!({"tools": [{"type": "mcp"}]})).is_err());
    assert!(
        reject_unsupported_generation_options(&json!({"response_format": {"type": "text"}}))
            .is_ok()
    );
    assert!(
        reject_unsupported_generation_options(&json!({"response_format": {"type": "json_object"}}))
            .is_err()
    );
}

#[test]
fn tool_aware_generation_validation_accepts_mcp_and_function_tools() {
    assert!(
        reject_unsupported_generation_options_with_tools(&json!({
            "tools": [{
                "type": "mcp",
                "server_id": "search",
                "name": "lookup"
            }],
            "tool_choice": "auto"
        }))
        .is_ok()
    );
    assert!(
        reject_unsupported_generation_options_with_tools(&json!({
            "tools": [{
                "type": "function",
                "function": {"name": "lookup", "description": "search docs"}
            }]
        }))
        .is_ok()
    );
    assert!(
        reject_unsupported_generation_options_with_tools(&json!({
            "functions": [{"name": "lookup", "description": "search docs"}]
        }))
        .is_ok()
    );
    assert!(
        reject_unsupported_generation_options_with_tools(&json!({
            "tools": [{"type": "file_search"}]
        }))
        .is_err()
    );
    assert!(
        reject_unsupported_generation_options_with_tools(&json!({
            "tools": [{"type": "mcp"}],
            "tool_choice": 7
        }))
        .is_err()
    );
}

#[test]
fn function_tools_are_prompt_context_not_mcp_invocations() {
    assert!(
        mcp_tool_invocation_from_request(&json!({
            "tools": [{
                "type": "function",
                "function": {"name": "lookup"}
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup"}
            }
        }))
        .unwrap()
        .is_none()
    );

    let augmented = augment_prompt_with_mcp_tool(
        "User: answer this\nAssistant:".to_string(),
        None,
        &json!({
            "functions": [{
                "name": "lookup",
                "description": "search docs",
                "parameters": {"type": "object"}
            }]
        }),
    );
    assert!(augmented.contains("Available tools, if needed:"));
    assert!(augmented.contains("function lookup: search docs"));
    assert!(augmented.ends_with("Assistant:"));
}

#[test]
fn parses_explicit_mcp_tool_invocation_from_request() {
    let invocation = mcp_tool_invocation_from_request(&json!({
        "tools": [{
            "type": "mcp",
            "server_id": "search",
            "name": "lookup"
        }],
        "tool_choice": {
            "type": "mcp",
            "name": "lookup",
            "arguments": "{\"query\":\"rust\"}"
        }
    }))
    .unwrap()
    .unwrap();

    assert_eq!(invocation.server_id.as_deref(), Some("search"));
    assert_eq!(invocation.name, "lookup");
    assert_eq!(invocation.arguments["query"], "rust");
}

#[test]
fn augments_prompt_with_mcp_tool_result_before_assistant_marker() {
    let prompt = "User: answer this\nAssistant:".to_string();
    let result = McpToolResult {
        server_id: "search".to_string(),
        name: "lookup".to_string(),
        arguments: json!({"query": "rust"}),
        result: json!({"answer": "Ferris"}),
    };

    let augmented = augment_prompt_with_mcp_tool(prompt, Some(&result), &json!({}));

    assert!(augmented.contains("MCP tool result from server 'search' tool 'lookup'"));
    assert!(augmented.contains("\"answer\":\"Ferris\""));
    assert!(augmented.ends_with("Assistant:"));
}

#[test]
fn parses_n_and_optional_strings() {
    assert_eq!(request_n(&json!({})).unwrap(), 1);
    assert_eq!(request_n(&json!({"n": 3})).unwrap(), 3);
    assert!(request_n(&json!({"n": 0})).is_err());
    assert_eq!(
        request_optional_string(&json!({"session_id": "abc"}), "session_id").unwrap(),
        Some("abc".to_string())
    );
    assert!(request_optional_string(&json!({"session_id": ""}), "session_id").is_err());
}

#[test]
fn hashes_tokens_stably() {
    let a = hash_tokens(&[
        nerva_core::types::id::token::TokenId(1),
        nerva_core::types::id::token::TokenId(2),
    ]);
    let b = hash_tokens(&[
        nerva_core::types::id::token::TokenId(1),
        nerva_core::types::id::token::TokenId(2),
    ]);
    let c = hash_tokens(&[
        nerva_core::types::id::token::TokenId(2),
        nerva_core::types::id::token::TokenId(1),
    ]);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn serializes_session_records() {
    let record = SessionRecord {
        id: "sess-1".to_string(),
        object: "session",
        created: 1,
        updated: 2,
        request_count: 3,
        prompt_tokens: 4,
        generated_tokens: 5,
        last_cache_key: Some("cache".to_string()),
        last_prompt_hash: Some(0xabcd),
    };
    let value = session_json(&record);
    assert_eq!(value["id"], "sess-1");
    assert_eq!(value["last_prompt_hash"], "000000000000abcd");
}

#[test]
fn gates_shared_fork_batch_to_greedy_sampler() {
    assert!(shared_fork_batch_supported(0.0, 1.0, 0, None));
    assert!(!shared_fork_batch_supported(1.0, 1.0, 0, None));
    assert!(!shared_fork_batch_supported(0.0, 0.95, 0, None));
    assert!(!shared_fork_batch_supported(0.0, 1.0, 40, None));
    assert!(!shared_fork_batch_supported(0.0, 1.0, 0, Some(7)));
}

#[test]
fn parses_multipart_file_upload() {
    let body = concat!(
        "--nerva\r\n",
        "Content-Disposition: form-data; name=\"purpose\"\r\n\r\n",
        "batch\r\n",
        "--nerva\r\n",
        "Content-Disposition: form-data; name=\"file\"; filename=\"requests.jsonl\"\r\n",
        "Content-Type: application/jsonl\r\n\r\n",
        "{\"custom_id\":\"a\"}\n",
        "\r\n--nerva--\r\n"
    );
    let upload = parse_multipart_file_upload(body.as_bytes(), "nerva").unwrap();

    assert_eq!(upload.filename, "requests.jsonl");
    assert_eq!(upload.purpose, "batch");
    assert_eq!(
        String::from_utf8(upload.content).unwrap(),
        "{\"custom_id\":\"a\"}\n"
    );
}

#[test]
fn normalizes_batch_endpoints() {
    assert_eq!(
        normalize_batch_endpoint("/v1/chat/completions").unwrap(),
        "/v1/chat/completions"
    );
    assert_eq!(
        normalize_batch_endpoint("http://localhost:8080/v1/responses").unwrap(),
        "/v1/responses"
    );
    assert!(normalize_batch_endpoint("/v1/embeddings").is_err());
}

#[test]
fn parses_mcp_http_targets_and_payloads() {
    let endpoint = parse_http_endpoint("http://127.0.0.1:9000/mcp").unwrap();
    assert_eq!(endpoint.host, "127.0.0.1");
    assert_eq!(endpoint.port, 9000);
    assert_eq!(endpoint.path, "/mcp");

    let response = b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"tools\":[]}}";
    let parsed = parse_mcp_http_response(response).unwrap();
    assert_eq!(parsed["tools"], json!([]));

    assert_eq!(
        first_sse_json_payload("event: message\ndata: {\"result\":true}\n\n").unwrap(),
        "{\"result\":true}"
    );
}

#[test]
fn decodes_chunked_mcp_body_and_query_escapes() {
    let decoded = decode_chunked_body(b"5\r\nhello\r\n0\r\n\r\n").unwrap();
    assert_eq!(decoded, b"hello");
    assert_eq!(percent_decode_query("hello+%77orld"), "hello world");
}
