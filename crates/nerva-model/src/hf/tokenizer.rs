use std::path::Path;

use nerva_core::types::id::token::TokenId;

mod bpe;
mod codec;
mod json;
mod prompt;
mod stop;

pub struct EncodedPrompt {
    pub input_mode: &'static str,
    pub token_ids: Vec<u32>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PromptFormat {
    Auto,
    Chat,
    DeepSeekChat,
    DeepSeekThinking,
    Raw,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FormattedPrompt {
    pub text: String,
    pub mode: &'static str,
}

pub fn format_prompt_for_model(
    path: &str,
    prompt: &str,
    format: PromptFormat,
) -> Result<FormattedPrompt, String> {
    prompt::format_prompt_for_model(path, prompt, format)
}

pub fn encode_text_prompt(path: &str, prompt: &str) -> Result<EncodedPrompt, String> {
    let tokenizer = codec::load_tokenizer(path)?;
    let input_mode = tokenizer.input_mode;
    let encoding = tokenizer
        .tokenizer
        .encode(prompt, false)
        .map_err(|err| format!("HF tokenizer encode failed: {err}"))?;
    if !encoding.get_ids().is_empty() {
        return Ok(EncodedPrompt {
            input_mode,
            token_ids: encoding.get_ids().to_vec(),
        });
    }
    if input_mode == "tokenizer_json" {
        let tokenizer = codec::load_bpe_file_tokenizer(path)?;
        let encoding = tokenizer
            .tokenizer
            .encode(prompt, false)
            .map_err(|err| format!("HF tokenizer BPE fallback encode failed: {err}"))?;
        if !encoding.get_ids().is_empty() {
            return Ok(EncodedPrompt {
                input_mode: tokenizer.input_mode,
                token_ids: encoding.get_ids().to_vec(),
            });
        }
    }
    Err("HF text prompt produced no tokens".to_string())
}

pub fn decode_generated_text(path: &str, tokens: &[TokenId]) -> Result<Option<String>, String> {
    if tokens.is_empty() {
        return Ok(Some(String::new()));
    }
    if !tokenizer_files_present(Path::new(path)) {
        return Ok(None);
    }
    let ids = tokens.iter().map(|token| token.0).collect::<Vec<_>>();
    let tokenizer = codec::load_tokenizer(path)?;
    let decoded = tokenizer.tokenizer.decode(&ids, true);
    let text = match decoded {
        Ok(text) if !text.is_empty() => text,
        Ok(text) if tokenizer.input_mode != "tokenizer_json" => text,
        Ok(_) | Err(_) => codec::load_bpe_file_tokenizer(path)?
            .tokenizer
            .decode(&ids, true)
            .map_err(|err| format!("HF tokenizer BPE fallback decode failed: {err}"))?,
    };
    Ok(Some(text))
}

pub fn stop_token_ids(path: &str) -> Result<Vec<u32>, String> {
    stop::stop_token_ids(path)
}

fn tokenizer_files_present(dir: &Path) -> bool {
    dir.join("tokenizer.json").is_file()
        || (dir.join("vocab.json").is_file() && dir.join("merges.txt").is_file())
}
