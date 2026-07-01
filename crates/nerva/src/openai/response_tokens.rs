use actix_web::{HttpRequest, HttpResponse, web};
use nerva_model::hf::tokenizer::{encode_text_prompt, format_prompt_for_model};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, PromptInput, apply_response_format_instruction,
    augment_prompt_with_mcp_tool, authorize, conversation_context, previous_response_context,
    request_conversation_id, request_optional_string, request_reasoning_mode,
    request_response_format_instruction, require_known_model, responses_prompt_for_reasoning,
};

pub(crate) async fn count_response_input_tokens(
    state: web::Data<AppState>,
    request: HttpRequest,
    body: web::Json<Value>,
) -> HttpResponse {
    let response = async {
        authorize(&state, &request)?;
        require_known_model(&state, &body)?;
        let prompt = response_count_prompt(&state, &body)?;
        let input_tokens = prompt_input_token_count(&state, &prompt)?;
        Ok::<_, ApiError>(HttpResponse::Ok().json(json!({
            "object": "response.input_tokens",
            "input_tokens": input_tokens
        })))
    }
    .await;
    response.unwrap_or_else(ApiError::into_response)
}

fn response_count_prompt(state: &AppState, body: &Value) -> Result<PromptInput, ApiError> {
    let reasoning_mode = request_reasoning_mode(state, body)?;
    let response_format_instruction = request_response_format_instruction(body)?;
    let conversation_id = request_conversation_id(body)?;
    let previous_response_id = request_optional_string(body, "previous_response_id")?;
    let conversation_context = conversation_context(state, conversation_id.as_deref())?;
    let previous_context = previous_response_context(state, previous_response_id.as_deref())?;
    let mut prompt = apply_response_format_instruction(
        match responses_prompt_for_reasoning(body, reasoning_mode)? {
            PromptInput::Text { text, format } => PromptInput::Text {
                text: augment_prompt_with_mcp_tool(text, None, body),
                format,
            },
            prompt => prompt,
        },
        response_format_instruction.as_deref(),
    )?;
    if let Some(previous_context) = previous_context {
        prompt = prepend_prompt_text(prompt, &previous_context);
    }
    if let Some(conversation_context) = conversation_context {
        prompt = prepend_prompt_text(prompt, &conversation_context);
    }
    Ok(prompt)
}

fn prompt_input_token_count(state: &AppState, prompt: &PromptInput) -> Result<usize, ApiError> {
    match prompt {
        PromptInput::Text { text, format } => {
            let formatted = format_prompt_for_model(&state.config.model_path, text, *format)
                .map_err(ApiError::bad_request)?;
            let encoded = encode_text_prompt(&state.config.model_path, &formatted.text)
                .map_err(ApiError::bad_request)?;
            Ok(encoded.token_ids.len())
        }
        PromptInput::TokenIds(tokens) => Ok(tokens.len()),
    }
}

fn prepend_prompt_text(prompt: PromptInput, prefix: &str) -> PromptInput {
    match prompt {
        PromptInput::Text { text, format } => PromptInput::Text {
            text: format!("{prefix}{text}"),
            format,
        },
        prompt => prompt,
    }
}
