use nerva_core::types::error::{NervaError, Result};

use crate::common::json::parse::{
    find_json_value_end, find_top_level_json_value, parse_json_string_at, parse_json_string_value,
    skip_json_ws,
};

pub(crate) fn required_usize_array(source: &str, key: &'static str) -> Result<Vec<usize>> {
    let value =
        find_top_level_json_value(source, key)?.ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("JSON object is missing required array {key}"),
        })?;
    parse_usize_array_value(value, key)
}

fn parse_usize_array_value(value: &str, key: &'static str) -> Result<Vec<usize>> {
    let value = value.trim();
    if !value.starts_with('[') || !value.ends_with(']') {
        return Err(NervaError::InvalidArgument {
            reason: format!("JSON field {key} must be an unsigned integer array"),
        });
    }
    let inner = value[1..value.len() - 1].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|part| {
            let part = part.trim();
            if part.starts_with('-') || part.is_empty() {
                return Err(NervaError::InvalidArgument {
                    reason: format!("JSON field {key} must contain unsigned integers"),
                });
            }
            let parsed = part
                .parse::<u64>()
                .map_err(|_| NervaError::InvalidArgument {
                    reason: format!("JSON field {key} must contain unsigned integers"),
                })?;
            usize::try_from(parsed).map_err(|_| NervaError::InvalidArgument {
                reason: format!("JSON field {key} value does not fit usize"),
            })
        })
        .collect()
}

pub(crate) fn parse_json_string_map_value(
    value: &str,
    key: &'static str,
) -> Result<Vec<(String, String)>> {
    let value = value.trim();
    let bytes = value.as_bytes();
    let mut index = skip_json_ws(bytes, 0);
    if index >= bytes.len() || bytes[index] != b'{' {
        return Err(NervaError::InvalidArgument {
            reason: format!("JSON field {key} must be an object"),
        });
    }
    index += 1;
    let mut entries = Vec::new();
    loop {
        index = skip_json_ws(bytes, index);
        if index >= bytes.len() {
            return Err(NervaError::InvalidArgument {
                reason: format!("JSON field {key} object is not closed"),
            });
        }
        if bytes[index] == b'}' {
            return Ok(entries);
        }
        if bytes[index] == b',' {
            index += 1;
            continue;
        }
        if bytes[index] != b'"' {
            return Err(NervaError::InvalidArgument {
                reason: format!("JSON field {key} object key must be a string"),
            });
        }
        let (field, after_key) = parse_json_string_at(value, index)?;
        index = skip_json_ws(bytes, after_key);
        if index >= bytes.len() || bytes[index] != b':' {
            return Err(NervaError::InvalidArgument {
                reason: format!("JSON field {key} object key is missing ':'"),
            });
        }
        index = skip_json_ws(bytes, index + 1);
        let value_start = index;
        let value_end = find_json_value_end(value, value_start)?;
        let mapped = parse_json_string_value(&value[value_start..value_end])?;
        entries.push((field, mapped));
        index = value_end;
    }
}
