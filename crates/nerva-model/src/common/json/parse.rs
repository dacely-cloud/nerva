use nerva_core::types::error::{NervaError, Result};

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
