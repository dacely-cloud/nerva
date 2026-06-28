use nerva_core::types::id::token::TokenId;

use crate::json::json_escape;

pub(crate) fn generated_text_json(path: &str, tokens: &[TokenId]) -> Result<String, String> {
    let Some(text) = nerva_model::hf::tokenizer::decode_generated_text(path, tokens)? else {
        return Ok("null".to_string());
    };
    Ok(format!("\"{}\"", json_escape(&text)))
}
