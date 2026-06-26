use nerva_core::types::{NervaError, Result};

pub(crate) fn required_usize(config_json: &str, key: &'static str) -> Result<usize> {
    optional_usize(config_json, key)?.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("HF config is missing required field {key}"),
    })
}

pub(crate) fn optional_usize(config_json: &str, key: &'static str) -> Result<Option<usize>> {
    let Some(value) = find_top_level_json_value(config_json, key)? else {
        return Ok(None);
    };
    if value.starts_with('-') {
        return Err(NervaError::InvalidArgument {
            reason: format!("HF config field {key} must be unsigned"),
        });
    }
    let parsed = value
        .parse::<u64>()
        .map_err(|_| NervaError::InvalidArgument {
            reason: format!("HF config field {key} must be an integer"),
        })?;
    usize::try_from(parsed)
        .map(Some)
        .map_err(|_| NervaError::InvalidArgument {
            reason: format!("HF config field {key} does not fit usize"),
        })
}

pub(crate) fn optional_f32(config_json: &str, key: &'static str) -> Result<Option<f32>> {
    let Some(value) = find_top_level_json_value(config_json, key)? else {
        return Ok(None);
    };
    let parsed = value
        .parse::<f32>()
        .map_err(|_| NervaError::InvalidArgument {
            reason: format!("HF config field {key} must be a number"),
        })?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(NervaError::InvalidArgument {
            reason: format!("HF config field {key} must be positive and finite"),
        });
    }
    Ok(Some(parsed))
}

pub(crate) fn optional_string(config_json: &str, key: &'static str) -> Result<Option<String>> {
    let Some(value) = find_top_level_json_value(config_json, key)? else {
        return Ok(None);
    };
    parse_json_string_value(value).map(Some)
}

pub(crate) fn optional_bool(config_json: &str, key: &'static str) -> Result<Option<bool>> {
    let Some(value) = find_top_level_json_value(config_json, key)? else {
        return Ok(None);
    };
    match value.trim() {
        "true" => Ok(Some(true)),
        "false" => Ok(Some(false)),
        _ => Err(NervaError::InvalidArgument {
            reason: format!("HF config field {key} must be a boolean"),
        }),
    }
}

pub(crate) fn optional_first_string(
    config_json: &str,
    key: &'static str,
) -> Result<Option<String>> {
    let Some(value) = find_top_level_json_value(config_json, key)? else {
        return Ok(None);
    };
    let value = value.trim();
    if value.starts_with('"') {
        return parse_json_string_value(value).map(Some);
    }
    if !value.starts_with('[') {
        return Err(NervaError::InvalidArgument {
            reason: format!("HF config field {key} must be a string array"),
        });
    }
    let mut index = skip_json_ws(value.as_bytes(), 1);
    if index < value.len() && value.as_bytes()[index] == b']' {
        return Ok(None);
    }
    if index >= value.len() || value.as_bytes()[index] != b'"' {
        return Err(NervaError::InvalidArgument {
            reason: format!("HF config field {key} must contain string values"),
        });
    }
    let (parsed, after) = parse_json_string_at(value, index)?;
    index = skip_json_ws(value.as_bytes(), after);
    if index >= value.len() || !matches!(value.as_bytes()[index], b',' | b']') {
        return Err(NervaError::InvalidArgument {
            reason: format!("HF config field {key} has malformed string array"),
        });
    }
    Ok(Some(parsed))
}

pub(crate) fn find_top_level_json_value<'a>(source: &'a str, key: &str) -> Result<Option<&'a str>> {
    let bytes = source.as_bytes();
    let mut index = skip_json_ws(bytes, 0);
    if index >= bytes.len() || bytes[index] != b'{' {
        return Err(NervaError::InvalidArgument {
            reason: "HF config must be a JSON object".to_string(),
        });
    }
    index += 1;

    loop {
        index = skip_json_ws(bytes, index);
        if index >= bytes.len() {
            return Err(NervaError::InvalidArgument {
                reason: "HF config object is not closed".to_string(),
            });
        }
        if bytes[index] == b'}' {
            return Ok(None);
        }
        if bytes[index] == b',' {
            index += 1;
            continue;
        }
        if bytes[index] != b'"' {
            return Err(NervaError::InvalidArgument {
                reason: "HF config object key must be a JSON string".to_string(),
            });
        }

        let (field, after_key) = parse_json_string_at(source, index)?;
        index = skip_json_ws(bytes, after_key);
        if index >= bytes.len() || bytes[index] != b':' {
            return Err(NervaError::InvalidArgument {
                reason: "HF config object key is missing ':'".to_string(),
            });
        }
        index = skip_json_ws(bytes, index + 1);
        let value_start = index;
        let value_end = find_json_value_end(source, value_start)?;
        if field == key {
            return Ok(Some(source[value_start..value_end].trim()));
        }
        index = value_end;
    }
}

pub(crate) fn find_json_value_end(source: &str, start: usize) -> Result<usize> {
    let mut depth = 0u32;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in source[start..].char_indices() {
        let index = start + offset;
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '[' | '{' => depth = depth.saturating_add(1),
            ']' => {
                if depth == 0 {
                    return Err(NervaError::InvalidArgument {
                        reason: "HF config has unmatched ']'".to_string(),
                    });
                }
                depth -= 1;
            }
            '}' => {
                if depth == 0 {
                    return Ok(index);
                }
                depth -= 1;
            }
            ',' if depth == 0 => return Ok(index),
            _ => {}
        }
    }
    if depth == 0 && !in_string {
        Ok(source.len())
    } else {
        Err(NervaError::InvalidArgument {
            reason: "HF config value is not closed".to_string(),
        })
    }
}

pub(crate) fn parse_json_string_value(value: &str) -> Result<String> {
    let value = value.trim();
    if !value.starts_with('"') {
        return Err(NervaError::InvalidArgument {
            reason: "HF config field must be a JSON string".to_string(),
        });
    }
    let (parsed, after) = parse_json_string_at(value, 0)?;
    if !value[after..].trim().is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: "HF config string field has trailing data".to_string(),
        });
    }
    Ok(parsed)
}

pub(crate) fn parse_json_string_at(source: &str, start: usize) -> Result<(String, usize)> {
    if source.as_bytes().get(start) != Some(&b'"') {
        return Err(NervaError::InvalidArgument {
            reason: "expected JSON string".to_string(),
        });
    }
    let mut out = String::new();
    let mut chars = source[start + 1..].char_indices();
    while let Some((offset, ch)) = chars.next() {
        let index = start + 1 + offset;
        match ch {
            '"' => return Ok((out, index + 1)),
            '\\' => {
                let Some((_, escaped)) = chars.next() else {
                    return Err(NervaError::InvalidArgument {
                        reason: "JSON string escape is incomplete".to_string(),
                    });
                };
                match escaped {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'b' => out.push('\u{0008}'),
                    'f' => out.push('\u{000c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        let mut codepoint = 0u32;
                        for _ in 0..4 {
                            let Some((_, hex)) = chars.next() else {
                                return Err(NervaError::InvalidArgument {
                                    reason: "JSON unicode escape is incomplete".to_string(),
                                });
                            };
                            let Some(value) = hex.to_digit(16) else {
                                return Err(NervaError::InvalidArgument {
                                    reason: "JSON unicode escape has non-hex digit".to_string(),
                                });
                            };
                            codepoint = (codepoint << 4) | value;
                        }
                        let Some(decoded) = char::from_u32(codepoint) else {
                            return Err(NervaError::InvalidArgument {
                                reason: "JSON unicode escape is invalid".to_string(),
                            });
                        };
                        out.push(decoded);
                    }
                    _ => {
                        return Err(NervaError::InvalidArgument {
                            reason: "unsupported JSON string escape".to_string(),
                        });
                    }
                }
            }
            ch => out.push(ch),
        }
    }
    Err(NervaError::InvalidArgument {
        reason: "JSON string is not closed".to_string(),
    })
}

pub(crate) fn skip_json_ws(bytes: &[u8], mut index: usize) -> usize {
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    index
}

pub(crate) fn json_opt_str(value: Option<&str>) -> String {
    value.map_or_else(
        || "null".to_string(),
        |value| format!("\"{}\"", json_escape(value)),
    )
}

pub(crate) fn json_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

pub(crate) fn json_opt_f32(value: Option<f32>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

pub(crate) fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => escaped.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped
}
