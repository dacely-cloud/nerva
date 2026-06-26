pub(crate) fn json_opt_i32(value: Option<i32>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

pub(crate) fn json_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

pub(crate) fn json_opt_bool(value: Option<bool>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

pub(crate) fn json_opt_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

pub(crate) fn json_opt_str(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("\"{}\"", escape_json(value)),
        None => "null".to_string(),
    }
}

pub(crate) fn escape_json(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
