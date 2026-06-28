use std::path::Path;

use tokenizers::Tokenizer;

use crate::hf::tokenizer::bpe::load_byte_bpe_tokenizer;

pub(super) struct LoadedTokenizer {
    pub(super) tokenizer: Tokenizer,
    pub(super) input_mode: &'static str,
}

pub(super) fn load_tokenizer(path: &str) -> Result<LoadedTokenizer, String> {
    let dir = Path::new(path);
    let tokenizer_path = dir.join("tokenizer.json");
    if tokenizer_path.is_file() {
        match Tokenizer::from_file(&tokenizer_path) {
            Ok(tokenizer) => {
                return Ok(LoadedTokenizer {
                    tokenizer,
                    input_mode: "tokenizer_json",
                });
            }
            Err(tokenizer_json_error) => {
                let fallback = load_byte_bpe_tokenizer(dir).map_err(|fallback_error| {
                    format!(
                        "HF tokenizer load failed: {tokenizer_json_error}; byte-level BPE fallback failed: {fallback_error}"
                    )
                })?;
                return Ok(LoadedTokenizer {
                    tokenizer: fallback,
                    input_mode: "tokenizer_bpe_files",
                });
            }
        }
    }
    let fallback = load_byte_bpe_tokenizer(dir)
        .map_err(|err| format!("HF tokenizer files not found or invalid: {err}"))?;
    Ok(LoadedTokenizer {
        tokenizer: fallback,
        input_mode: "tokenizer_bpe_files",
    })
}

pub(super) fn load_bpe_file_tokenizer(path: &str) -> Result<LoadedTokenizer, String> {
    let tokenizer = load_byte_bpe_tokenizer(Path::new(path))
        .map_err(|err| format!("HF byte-level BPE tokenizer load failed: {err}"))?;
    Ok(LoadedTokenizer {
        tokenizer,
        input_mode: "tokenizer_bpe_files",
    })
}
