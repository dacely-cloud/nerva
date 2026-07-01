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
    let dir = Path::new(path);
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
    if input_mode == "tokenizer_json" && bpe_files_present(dir) {
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
    let text = match tokenizer.tokenizer.decode(&ids, true) {
        Ok(text) => text,
        Err(err)
            if tokenizer.input_mode == "tokenizer_json" && bpe_files_present(Path::new(path)) =>
        {
            codec::load_bpe_file_tokenizer(path)?
                .tokenizer
                .decode(&ids, true)
                .map_err(|err| format!("HF tokenizer BPE fallback decode failed: {err}"))?
        }
        Err(err) => return Err(format!("HF tokenizer decode failed: {err}")),
    };
    Ok(Some(text))
}

pub fn stop_token_ids(path: &str) -> Result<Vec<u32>, String> {
    stop::stop_token_ids(path)
}

fn tokenizer_files_present(dir: &Path) -> bool {
    dir.join("tokenizer.json").is_file() || bpe_files_present(dir)
}

fn bpe_files_present(dir: &Path) -> bool {
    dir.join("vocab.json").is_file() && dir.join("merges.txt").is_file()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use nerva_core::types::id::token::TokenId;

    use super::{decode_generated_text, encode_text_prompt};

    #[test]
    fn tokenizer_json_only_decode_all_special_tokens_returns_empty_text() {
        let dir = tokenizer_json_dir("decode-empty-specials");

        let text = decode_generated_text(dir.to_str().unwrap(), &[TokenId(0)]).unwrap();
        assert_eq!(text.as_deref(), Some(""));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn tokenizer_json_only_encode_does_not_require_bpe_sidecar_files() {
        let dir = tokenizer_json_dir("encode-without-bpe-sidecars");

        let encoded = encode_text_prompt(dir.to_str().unwrap(), "hello").unwrap();
        assert_eq!(encoded.input_mode, "tokenizer_json");
        assert_eq!(encoded.token_ids, vec![1]);

        let _ = fs::remove_dir_all(dir);
    }

    fn tokenizer_json_dir(name: &str) -> std::path::PathBuf {
        let dir = temp_dir(name);
        fs::write(
            dir.join("tokenizer.json"),
            r#"{
              "version": "1.0",
              "truncation": null,
              "padding": null,
              "added_tokens": [
                {
                  "id": 0,
                  "content": "<pad>",
                  "single_word": false,
                  "lstrip": false,
                  "rstrip": false,
                  "normalized": false,
                  "special": true
                }
              ],
              "normalizer": null,
              "pre_tokenizer": null,
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": {
                  "<pad>": 0,
                  "hello": 1
                },
                "unk_token": "<pad>"
              }
            }"#,
        )
        .unwrap();
        dir
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("nerva-tokenizer-{name}-{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
