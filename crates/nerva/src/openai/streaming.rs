use actix_web::web;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use super::{ApiError, ReasoningMode, StreamKind, StreamMeta, split_generated_reasoning, usage};

#[derive(Default)]
pub(crate) struct StreamEmissionState {
    content: String,
    reasoning: String,
    response_reasoning_started: bool,
}

pub(crate) fn emit_stream_text(
    tx: &mpsc::Sender<web::Bytes>,
    kind: StreamKind,
    meta: &StreamMeta,
    include_reasoning: bool,
    reasoning_mode: ReasoningMode,
    emitted: &mut StreamEmissionState,
    current_text: &str,
) -> bool {
    let split = split_generated_reasoning(current_text, reasoning_mode);
    if include_reasoning {
        let reasoning_delta = text_delta(&emitted.reasoning, &split.reasoning);
        if !reasoning_delta.is_empty()
            && !send_stream_reasoning_delta(
                tx,
                kind,
                meta,
                reasoning_delta,
                &mut emitted.response_reasoning_started,
            )
        {
            return false;
        }
        emitted.reasoning = split.reasoning;
    }

    let content_delta = text_delta(&emitted.content, &split.content);
    if !content_delta.is_empty() && !send_stream_delta(tx, kind, meta, content_delta) {
        return false;
    }
    emitted.content = split.content;
    true
}

pub(crate) fn send_stream_reasoning_delta(
    tx: &mpsc::Sender<web::Bytes>,
    kind: StreamKind,
    meta: &StreamMeta,
    delta: &str,
    response_reasoning_started: &mut bool,
) -> bool {
    match kind {
        StreamKind::Completion => true,
        StreamKind::ChatCompletion => send_sse_json(
            tx,
            None,
            json!({
                "id": meta.id,
                "object": "chat.completion.chunk",
                "created": meta.created,
                "model": meta.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning": delta,
                        "reasoning_content": delta
                    },
                    "logprobs": null,
                    "finish_reason": null
                }]
            }),
        ),
        StreamKind::Response => {
            if !*response_reasoning_started {
                *response_reasoning_started = true;
                if !send_sse_json(
                    tx,
                    Some("response.output_item.added"),
                    json!({
                        "type": "response.output_item.added",
                        "response_id": meta.id,
                        "output_index": 0,
                        "item": {
                            "id": format!("{}-reasoning", meta.id),
                            "type": "reasoning",
                            "summary": [],
                            "status": "in_progress"
                        }
                    }),
                ) {
                    return false;
                }
                if !send_sse_json(
                    tx,
                    Some("response.reasoning_part.added"),
                    json!({
                        "type": "response.reasoning_part.added",
                        "response_id": meta.id,
                        "item_id": format!("{}-reasoning", meta.id),
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "reasoning_text",
                            "text": ""
                        }
                    }),
                ) {
                    return false;
                }
            }
            send_sse_json(
                tx,
                Some("response.reasoning_text.delta"),
                json!({
                    "type": "response.reasoning_text.delta",
                    "response_id": meta.id,
                    "item_id": format!("{}-reasoning", meta.id),
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta
                }),
            )
        }
    }
}

fn send_stream_delta(
    tx: &mpsc::Sender<web::Bytes>,
    kind: StreamKind,
    meta: &StreamMeta,
    delta: &str,
) -> bool {
    match kind {
        StreamKind::Completion => send_sse_json(
            tx,
            None,
            json!({
                "id": meta.id,
                "object": "text_completion",
                "created": meta.created,
                "model": meta.model,
                "choices": [{
                    "text": delta,
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": null
                }]
            }),
        ),
        StreamKind::ChatCompletion => send_sse_json(
            tx,
            None,
            json!({
                "id": meta.id,
                "object": "chat.completion.chunk",
                "created": meta.created,
                "model": meta.model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": delta
                    },
                    "logprobs": null,
                    "finish_reason": null
                }]
            }),
        ),
        StreamKind::Response => send_sse_json(
            tx,
            Some("response.output_text.delta"),
            json!({
                "type": "response.output_text.delta",
                "response_id": meta.id,
                "item_id": format!("{}-message", meta.id),
                "output_index": 0,
                "content_index": 0,
                "delta": delta
            }),
        ),
    }
}

pub(crate) fn send_stream_final(
    tx: &mpsc::Sender<web::Bytes>,
    kind: StreamKind,
    meta: &StreamMeta,
    finish_reason: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    completed_response: Option<Value>,
) -> bool {
    match kind {
        StreamKind::Completion => send_sse_json(
            tx,
            None,
            json!({
                "id": meta.id,
                "object": "text_completion",
                "created": meta.created,
                "model": meta.model,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": finish_reason
                }],
                "usage": usage(prompt_tokens, completion_tokens)
            }),
        ),
        StreamKind::ChatCompletion => send_sse_json(
            tx,
            None,
            json!({
                "id": meta.id,
                "object": "chat.completion.chunk",
                "created": meta.created,
                "model": meta.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "logprobs": null,
                    "finish_reason": finish_reason
                }],
                "usage": usage(prompt_tokens, completion_tokens)
            }),
        ),
        StreamKind::Response => {
            let response = completed_response.unwrap_or_else(|| {
                json!({
                    "id": meta.id,
                    "object": "response",
                    "created_at": meta.created,
                    "status": "completed",
                    "error": null,
                    "incomplete_details": null,
                    "model": meta.model,
                    "metadata": null,
                    "store": false,
                    "previous_response_id": null,
                    "output": [],
                    "output_text": "",
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                })
            });
            send_sse_json(
                tx,
                Some("response.completed"),
                json!({
                    "type": "response.completed",
                    "response": response
                }),
            )
        }
    }
}

pub(crate) fn response_stream_completed_response(
    meta: &StreamMeta,
    emitted: &StreamEmissionState,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> Option<Value> {
    let response_options = meta.response.as_ref()?;
    let mut output = Vec::new();
    if !emitted.reasoning.is_empty() {
        output.push(json!({
            "id": format!("{}-reasoning", meta.id),
            "type": "reasoning",
            "summary": [],
            "status": "completed",
            "content": [{
                "type": "reasoning_text",
                "text": emitted.reasoning
            }]
        }));
    }
    output.push(json!({
        "id": format!("{}-message", meta.id),
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{
            "id": format!("{}-content", meta.id),
            "type": "output_text",
            "text": emitted.content,
            "annotations": []
        }]
    }));
    Some(json!({
        "id": meta.id,
        "object": "response",
        "created_at": meta.created,
        "status": "completed",
        "error": null,
        "incomplete_details": null,
        "model": meta.model,
        "metadata": response_options.metadata.clone(),
        "store": response_options.store,
        "previous_response_id": response_options.previous_response_id.clone(),
        "conversation": response_options.conversation_id.clone(),
        "output": output,
        "output_text": emitted.content,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }))
}

pub(crate) fn send_stream_error(tx: &mpsc::Sender<web::Bytes>, error: ApiError) {
    let _ = send_sse_json(
        tx,
        Some("error"),
        json!({
            "error": {
                "message": error.message,
                "type": error.code,
                "param": null,
                "code": error.code
            }
        }),
    );
    send_stream_done(tx);
}

pub(crate) fn send_stream_done(tx: &mpsc::Sender<web::Bytes>) -> bool {
    tx.blocking_send(web::Bytes::from_static(b"data: [DONE]\n\n"))
        .is_ok()
}

fn send_sse_json(tx: &mpsc::Sender<web::Bytes>, event: Option<&str>, value: Value) -> bool {
    let mut frame = String::new();
    if let Some(event) = event {
        frame.push_str("event: ");
        frame.push_str(event);
        frame.push('\n');
    }
    frame.push_str("data: ");
    frame.push_str(&value.to_string());
    frame.push_str("\n\n");
    tx.blocking_send(web::Bytes::from(frame)).is_ok()
}

pub(crate) fn text_delta<'a>(previous: &str, current: &'a str) -> &'a str {
    current.strip_prefix(previous).unwrap_or(current)
}
