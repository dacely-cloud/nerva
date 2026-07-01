use nerva_core::types::id::token::TokenId;

const TOKEN_KEYS: &[&str] = &[
    "token_ids",
    "output_token_ids",
    "tokens",
    "generated_token_ids",
];

pub(crate) fn parse_vllm_token_ids(source: &str) -> Result<(&'static str, Vec<TokenId>), String> {
    for key in TOKEN_KEYS {
        if let Some(value_start) = find_json_array_for_key(source, key)? {
            return parse_token_array(source, value_start).map(|tokens| (*key, tokens));
        }
    }
    Err(format!(
        "vLLM token JSON does not contain any supported token id key: {}",
        TOKEN_KEYS.join(", ")
    ))
}

pub(crate) fn parse_token_ids_for_key(
    source: &str,
    key: &str,
) -> Result<Option<Vec<TokenId>>, String> {
    let Some(value_start) = find_json_array_for_key(source, key)? else {
        return Ok(None);
    };
    parse_token_array(source, value_start).map(Some)
}

fn find_json_array_for_key(source: &str, key: &str) -> Result<Option<usize>, String> {
    let bytes = source.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] != b'"' {
            index += 1;
            continue;
        }
        let (field, after_field) = parse_json_string(source, index)?;
        index = after_field;
        if field != key {
            continue;
        }
        let colon = skip_ws(bytes, after_field);
        if bytes.get(colon) != Some(&b':') {
            continue;
        }
        let value_start = skip_ws(bytes, colon + 1);
        if bytes.get(value_start) == Some(&b'[') {
            return Ok(Some(value_start));
        }
    }
    Ok(None)
}

fn parse_token_array(source: &str, start: usize) -> Result<Vec<TokenId>, String> {
    let bytes = source.as_bytes();
    if bytes.get(start) != Some(&b'[') {
        return Err("token id value must be an array".to_string());
    }
    let mut index = start + 1;
    let mut tokens = Vec::new();
    loop {
        index = skip_ws(bytes, index);
        match bytes.get(index) {
            Some(b']') => return Ok(tokens),
            Some(b',') => {
                index += 1;
                continue;
            }
            Some(b'-') => return Err("token ids must be unsigned integers".to_string()),
            Some(b'0'..=b'9') => {
                let (value, next) = parse_u32(source, index)?;
                tokens.push(TokenId(value));
                index = next;
            }
            Some(_) => return Err("token id array contains a non-integer value".to_string()),
            None => return Err("token id array is not closed".to_string()),
        }
    }
}

fn parse_u32(source: &str, start: usize) -> Result<(u32, usize), String> {
    let bytes = source.as_bytes();
    let mut end = start;
    while matches!(bytes.get(end), Some(b'0'..=b'9')) {
        end += 1;
    }
    let value = source[start..end]
        .parse::<u64>()
        .map_err(|_| "token id is not a valid integer".to_string())?;
    let value = u32::try_from(value).map_err(|_| "token id does not fit u32".to_string())?;
    Ok((value, end))
}

fn parse_json_string(source: &str, start: usize) -> Result<(String, usize), String> {
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
                    'b' => out.push('\u{0008}'),
                    'f' => out.push('\u{000c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        let mut codepoint = 0u32;
                        for _ in 0..4 {
                            let Some((_, hex)) = chars.next() else {
                                return Err("JSON unicode escape is incomplete".to_string());
                            };
                            let Some(value) = hex.to_digit(16) else {
                                return Err("JSON unicode escape has non-hex digit".to_string());
                            };
                            codepoint = (codepoint << 4) | value;
                        }
                        let Some(decoded) = char::from_u32(codepoint) else {
                            return Err("JSON unicode escape is invalid".to_string());
                        };
                        out.push(decoded);
                    }
                    _ => return Err("unsupported JSON string escape".to_string()),
                }
            }
            ch => out.push(ch),
        }
    }
    Err("JSON string is not closed".to_string())
}

fn skip_ws(bytes: &[u8], mut index: usize) -> usize {
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    index
}
