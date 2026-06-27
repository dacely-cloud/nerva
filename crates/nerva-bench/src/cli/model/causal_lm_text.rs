use std::path::Path;

use nerva_core::types::id::token::TokenId;
use tokenizers::Tokenizer;

use crate::json::json_escape;

pub(crate) fn generated_text_json(path: &str, tokens: &[TokenId]) -> Result<String, String> {
    let tokenizer_path = Path::new(path).join("tokenizer.json");
    if !tokenizer_path.is_file() {
        return Ok("null".to_string());
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|err| format!("HF tokenizer load failed: {err}"))?;
    let ids = tokens.iter().map(|token| token.0).collect::<Vec<_>>();
    let text = tokenizer
        .decode(&ids, true)
        .map_err(|err| format!("HF tokenizer decode failed: {err}"))?;
    Ok(format!("\"{}\"", json_escape(&text)))
}
