use std::collections::BTreeSet;
use std::path::Path;

use serde_json::Value;

use crate::hf::tokenizer::json::{read_json_file, token_content};

pub(super) fn stop_token_ids(path: &str) -> Result<Vec<u32>, String> {
    let dir = Path::new(path);
    let mut ids = BTreeSet::new();
    if let Some(config) = read_json_file(&dir.join("config.json"))? {
        collect_config_eos_tokens(&config, &mut ids)?;
    }
    if let Some(tokenizer_config) = read_json_file(&dir.join("tokenizer_config.json"))? {
        if let Some(content) = tokenizer_config.get("eos_token").and_then(token_content) {
            collect_added_token_id(&tokenizer_config, content, &mut ids)?;
        }
        for content in common_chat_stop_token_contents() {
            collect_added_token_id(&tokenizer_config, content, &mut ids)?;
        }
    }
    Ok(ids.into_iter().collect())
}

fn collect_config_eos_tokens(config: &Value, ids: &mut BTreeSet<u32>) -> Result<(), String> {
    let Some(value) = config.get("eos_token_id") else {
        return Ok(());
    };
    match value {
        Value::Number(number) => {
            let Some(id) = number.as_u64().and_then(|id| u32::try_from(id).ok()) else {
                return Err("config.json eos_token_id must fit u32".to_string());
            };
            ids.insert(id);
        }
        Value::Array(values) => {
            for value in values {
                let Some(id) = value.as_u64().and_then(|id| u32::try_from(id).ok()) else {
                    return Err(
                        "config.json eos_token_id array must contain u32 values".to_string()
                    );
                };
                ids.insert(id);
            }
        }
        Value::Null => {}
        _ => {
            return Err(
                "config.json eos_token_id must be a token id or token id array".to_string(),
            );
        }
    }
    Ok(())
}

fn collect_added_token_id(
    tokenizer_config: &Value,
    content: &str,
    ids: &mut BTreeSet<u32>,
) -> Result<(), String> {
    let Some(decoder) = tokenizer_config
        .get("added_tokens_decoder")
        .and_then(Value::as_object)
    else {
        return Ok(());
    };
    for (id, token) in decoder {
        if token_content(token) == Some(content) {
            let parsed = id
                .parse::<u32>()
                .map_err(|_| "tokenizer_config added token id must fit u32".to_string())?;
            ids.insert(parsed);
        }
    }
    Ok(())
}

fn common_chat_stop_token_contents() -> impl Iterator<Item = &'static str> {
    ["<|im_end|>", "<|eot_id|>", "<end_of_turn>"].into_iter()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::hf::tokenizer::stop_token_ids;

    #[test]
    fn reads_config_and_chat_stop_tokens() {
        let dir = temp_dir("stop-tokens");
        fs::write(dir.join("config.json"), r#"{"eos_token_id":[2,3]}"#).unwrap();
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"eos_token":"<|im_end|>","added_tokens_decoder":{"151645":{"content":"<|im_end|>"}}}"#,
        )
        .unwrap();
        assert_eq!(
            stop_token_ids(dir.to_str().unwrap()).unwrap(),
            vec![2, 3, 151645]
        );
        let _ = fs::remove_dir_all(dir);
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
