use std::sync::atomic::Ordering;

use actix_web::{HttpRequest, HttpResponse, web};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, GenerateOptions, PromptInput, ReasoningMode, StreamKind, StreamMeta,
    augment_prompt_with_mcp_tool, authorize, chat_messages_to_prompt, completion_prompts,
    empty_text_prompt, execute_request_mcp_tool, generate_text, generate_text_batch,
    generate_text_stream, mcp_tool_result_json, prompt_format_for_reasoning,
    reject_unsupported_generation_options, reject_unsupported_generation_options_with_tools,
    request_f32, request_include_reasoning, request_max_tokens, request_n, request_optional_string,
    request_reasoning_mode, request_stop_strings, request_stream, request_u32, request_u64_opt,
    require_known_model, responses_input_to_prompt, shared_fork_batch_supported,
    split_generated_reasoning, unix_seconds, usage,
};

pub(crate) async fn completions(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        require_known_model(&state, &body)?;
        reject_unsupported_generation_options(&body)?;
        let n = request_n(&body)?;
        let prompts = completion_prompts(&body)?;
        let max_tokens = request_max_tokens(&state, &body)?;
        let temperature = request_f32(&body, "temperature", 1.0)?;
        let top_p = request_f32(&body, "top_p", 1.0)?;
        let top_k = request_u32(&body, "top_k", 0)?;
        let seed = request_u64_opt(&body, "seed")?;
        let stop = request_stop_strings(&body)?;
        let session_id = request_optional_string(&body, "session_id")?;
        let cache_key = request_optional_string(&body, "cache_key")?;
        let created = unix_seconds();
        let id = state.next_response_id("cmpl");
        if request_stream(&body) {
            if prompts.len() != 1 || n != 1 {
                return Err(ApiError::unsupported(
                    "streaming completions currently require exactly one prompt and n=1",
                ));
            }
            return generate_text_stream(
                state.clone(),
                GenerateOptions {
                    prompt: prompts.into_iter().next().unwrap_or_else(empty_text_prompt),
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    seed,
                    stop,
                    session_id,
                    cache_key,
                    include_reasoning: false,
                    reasoning_mode: ReasoningMode::None,
                },
                StreamKind::Completion,
                StreamMeta {
                    id: id.clone(),
                    created,
                    model: state.config.model_id.clone(),
                },
            )
            .await;
        }
        let mut choices = Vec::with_capacity(prompts.len().saturating_mul(n));
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        if n > 1
            && prompts.len() == 1
            && shared_fork_batch_supported(temperature, top_p, top_k, seed)
        {
            let generated = generate_text_batch(
                state.clone(),
                GenerateOptions {
                    prompt: prompts.into_iter().next().unwrap_or_else(empty_text_prompt),
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    seed,
                    stop: stop.clone(),
                    session_id: session_id.clone(),
                    cache_key: cache_key.clone(),
                    include_reasoning: false,
                    reasoning_mode: ReasoningMode::None,
                },
                n,
            )
            .await?;
            for item in generated {
                prompt_tokens += item.prompt_tokens;
                completion_tokens += item.token_ids.len();
                choices.push(json!({
                    "text": item.text,
                    "index": choices.len(),
                    "logprobs": null,
                    "finish_reason": item.finish_reason
                }));
            }
        } else {
            for prompt in prompts {
                for _ in 0..n {
                    let index = choices.len();
                    let generated = generate_text(
                        state.clone(),
                        GenerateOptions {
                            prompt: prompt.clone(),
                            max_tokens,
                            temperature,
                            top_p,
                            top_k,
                            seed,
                            stop: stop.clone(),
                            session_id: session_id.clone(),
                            cache_key: cache_key.clone(),
                            include_reasoning: false,
                            reasoning_mode: ReasoningMode::None,
                        },
                    )
                    .await?;
                    prompt_tokens += generated.prompt_tokens;
                    completion_tokens += generated.token_ids.len();
                    choices.push(json!({
                        "text": generated.text,
                        "index": index,
                        "logprobs": null,
                        "finish_reason": generated.finish_reason
                    }));
                }
            }
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "text_completion",
            "created": created,
            "model": state.config.model_id,
            "choices": choices,
            "usage": usage(prompt_tokens, completion_tokens)
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn chat_completions(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        require_known_model(&state, &body)?;
        reject_unsupported_generation_options_with_tools(&body)?;
        let n = request_n(&body)?;
        let tool_result = execute_request_mcp_tool(state.clone(), &body).await?;
        let prompt = augment_prompt_with_mcp_tool(
            chat_messages_to_prompt(&body)?,
            tool_result.as_ref(),
            &body,
        );
        let created = unix_seconds();
        let id = state.next_response_id("chatcmpl");
        let include_reasoning = request_include_reasoning(&body)?;
        let reasoning_mode = request_reasoning_mode(&state, &body)?;
        let prompt_format = prompt_format_for_reasoning(reasoning_mode);
        let options = GenerateOptions {
            prompt: PromptInput::Text {
                text: prompt,
                format: prompt_format,
            },
            max_tokens: request_max_tokens(&state, &body)?,
            temperature: request_f32(&body, "temperature", 1.0)?,
            top_p: request_f32(&body, "top_p", 1.0)?,
            top_k: request_u32(&body, "top_k", 0)?,
            seed: request_u64_opt(&body, "seed")?,
            stop: request_stop_strings(&body)?,
            session_id: request_optional_string(&body, "session_id")?,
            cache_key: request_optional_string(&body, "cache_key")?,
            include_reasoning,
            reasoning_mode,
        };
        if request_stream(&body) {
            if n != 1 {
                return Err(ApiError::unsupported(
                    "streaming chat completions currently require n=1",
                ));
            }
            return generate_text_stream(
                state.clone(),
                options,
                StreamKind::ChatCompletion,
                StreamMeta {
                    id: id.clone(),
                    created,
                    model: state.config.model_id.clone(),
                },
            )
            .await;
        }
        let response_include_reasoning = options.include_reasoning;
        let response_reasoning_mode = options.reasoning_mode;
        let mut choices = Vec::with_capacity(n);
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        for index in 0..n {
            let generated = generate_text(state.clone(), options.clone()).await?;
            let split = split_generated_reasoning(&generated.text, response_reasoning_mode);
            prompt_tokens += generated.prompt_tokens;
            completion_tokens += generated.token_ids.len();
            let mut message = json!({
                "role": "assistant",
                "content": split.content
            });
            if response_include_reasoning && !split.reasoning.is_empty() {
                message["reasoning"] = json!(split.reasoning);
                message["reasoning_content"] = json!(message["reasoning"].as_str().unwrap_or(""));
            }
            choices.push(json!({
                "index": index,
                "message": message,
                "logprobs": null,
                "finish_reason": generated.finish_reason
            }));
        }
        let mut response = json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": state.config.model_id,
            "choices": choices,
            "usage": usage(prompt_tokens, completion_tokens)
        });
        if let Some(tool_result) = tool_result {
            response["mcp_tool_results"] = json!([mcp_tool_result_json(&tool_result)]);
        }
        Ok::<_, ApiError>(HttpResponse::Ok().json(response))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

pub(crate) async fn responses(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    state.request_count.fetch_add(1, Ordering::Relaxed);
    let response = async {
        authorize(&state, &request)?;
        require_known_model(&state, &body)?;
        reject_unsupported_generation_options_with_tools(&body)?;
        let tool_result = execute_request_mcp_tool(state.clone(), &body).await?;
        let prompt = augment_prompt_with_mcp_tool(
            responses_input_to_prompt(&body)?,
            tool_result.as_ref(),
            &body,
        );
        let created = unix_seconds();
        let id = state.next_response_id("resp");
        let include_reasoning = request_include_reasoning(&body)?;
        let reasoning_mode = request_reasoning_mode(&state, &body)?;
        let prompt_format = prompt_format_for_reasoning(reasoning_mode);
        let options = GenerateOptions {
            prompt: PromptInput::Text {
                text: prompt,
                format: prompt_format,
            },
            max_tokens: request_max_tokens(&state, &body)?,
            temperature: request_f32(&body, "temperature", 1.0)?,
            top_p: request_f32(&body, "top_p", 1.0)?,
            top_k: request_u32(&body, "top_k", 0)?,
            seed: request_u64_opt(&body, "seed")?,
            stop: request_stop_strings(&body)?,
            session_id: request_optional_string(&body, "session_id")?,
            cache_key: request_optional_string(&body, "cache_key")?,
            include_reasoning,
            reasoning_mode,
        };
        if request_stream(&body) {
            return generate_text_stream(
                state.clone(),
                options,
                StreamKind::Response,
                StreamMeta {
                    id: id.clone(),
                    created,
                    model: state.config.model_id.clone(),
                },
            )
            .await;
        }
        let response_include_reasoning = options.include_reasoning;
        let response_reasoning_mode = options.reasoning_mode;
        let generated = generate_text(state.clone(), options).await?;
        let split = split_generated_reasoning(&generated.text, response_reasoning_mode);
        let output_id = state.next_response_id("msg");
        let content_id = state.next_response_id("ct");
        let completion_tokens = generated.token_ids.len();
        let mut output = Vec::new();
        if let Some(tool_result) = tool_result.as_ref() {
            output.push(json!({
                "id": state.next_response_id("mcp"),
                "type": "mcp_call",
                "status": "completed",
                "server_id": tool_result.server_id,
                "name": tool_result.name,
                "arguments": tool_result.arguments,
                "output": tool_result.result
            }));
        }
        if response_include_reasoning && !split.reasoning.is_empty() {
            output.push(json!({
                "id": state.next_response_id("rsn"),
                "type": "reasoning",
                "summary": [],
                "status": "completed",
                "content": [{
                    "type": "reasoning_text",
                    "text": split.reasoning
                }]
            }));
        }
        output.push(json!({
            "id": output_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "id": content_id,
                "type": "output_text",
                "text": split.content,
                "annotations": []
            }]
        }));
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "id": id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "model": state.config.model_id,
            "output": output,
            "output_text": split.content,
            "usage": {
                "input_tokens": generated.prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": generated.prompt_tokens + completion_tokens
            }
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}
