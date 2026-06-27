pub(crate) fn required_string(source: &str, key: &str) -> Result<String, String> {
    let value = required_value(source, key)?;
    parse_json_string_value(value).map_err(|err| format!("{key}: {err}"))
}

pub(crate) fn required_f64(source: &str, key: &str) -> Result<f64, String> {
    let value = required_value(source, key)?;
    value
        .parse::<f64>()
        .map_err(|_| format!("{key} must be a JSON number"))
}

pub(crate) fn optional_bool(source: &str, key: &str, default: bool) -> Result<bool, String> {
    match top_level_value(source, key)? {
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(_) => Err(format!("{key} must be a JSON boolean")),
        None => Ok(default),
    }
}

fn required_value<'a>(source: &'a str, key: &str) -> Result<&'a str, String> {
    top_level_value(source, key)?.ok_or_else(|| format!("{key} is required"))
}

fn top_level_value<'a>(source: &'a str, key: &str) -> Result<Option<&'a str>, String> {
    let bytes = source.as_bytes();
    let mut index = skip_ws(bytes, 0);
    if bytes.get(index) != Some(&b'{') {
        return Err("perf artifact must be a JSON object".to_string());
    }
    index += 1;

    loop {
        index = skip_ws(bytes, index);
        match bytes.get(index) {
            Some(b'}') => return Ok(None),
            Some(b',') => {
                index += 1;
                continue;
            }
            Some(b'"') => {}
            Some(_) => return Err("perf artifact key must be a JSON string".to_string()),
            None => return Err("perf artifact object is not closed".to_string()),
        }

        let (field, after_key) = parse_json_string_at(source, index)?;
        index = skip_ws(bytes, after_key);
        if bytes.get(index) != Some(&b':') {
            return Err("perf artifact key is missing ':'".to_string());
        }
        let value_start = skip_ws(bytes, index + 1);
        let value_end = json_value_end(source, value_start)?;
        if field == key {
            return Ok(Some(source[value_start..value_end].trim()));
        }
        index = value_end;
    }
}

fn json_value_end(source: &str, start: usize) -> Result<usize, String> {
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
            ']' | '}' if depth > 0 => depth -= 1,
            '}' => return Ok(index),
            ',' if depth == 0 => return Ok(index),
            _ => {}
        }
    }
    if depth == 0 && !in_string {
        Ok(source.len())
    } else {
        Err("perf artifact value is not closed".to_string())
    }
}

fn parse_json_string_value(value: &str) -> Result<String, String> {
    let value = value.trim();
    let (parsed, after) = parse_json_string_at(value, 0)?;
    if value[after..].trim().is_empty() {
        Ok(parsed)
    } else {
        Err("string field has trailing data".to_string())
    }
}

fn parse_json_string_at(source: &str, start: usize) -> Result<(String, usize), String> {
    if source.as_bytes().get(start) != Some(&b'"') {
        return Err("expected JSON string".to_string());
    }
    let mut out = String::new();
    let mut chars = source[start + 1..].char_indices();
    while let Some((offset, ch)) = chars.next() {
        let index = start + 1 + offset;
        match ch {
            '"' => return Ok((out, index + 1)),
            '\\' => {
                let Some((_, escaped)) = chars.next() else {
                    return Err("JSON string escape is incomplete".to_string());
                };
                match escaped {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    _ => return Err("unsupported JSON string escape".to_string()),
                }
            }
            ch => out.push(ch),
        }
    }
    Err("JSON string is not closed".to_string())
}

fn skip_ws(bytes: &[u8], mut index: usize) -> usize {
    while matches!(bytes.get(index), Some(b' ' | b'\n' | b'\r' | b'\t')) {
        index += 1;
    }
    index
}
