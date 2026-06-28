use std::fs;
use std::path::Path;

use serde_json::Value;

pub(super) fn read_json_file(path: &Path) -> Result<Option<Value>, String> {
    if !path.is_file() {
        return Ok(None);
    }
    let contents = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    serde_json::from_str(&contents)
        .map(Some)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

pub(super) fn token_content(value: &Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("content").and_then(Value::as_str))
}
