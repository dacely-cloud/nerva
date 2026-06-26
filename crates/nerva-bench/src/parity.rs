use std::fs;

use nerva_core::TokenId;

use crate::json::json_escape;

const TOKEN_KEYS: &[&str] = &[
    "token_ids",
    "output_token_ids",
    "tokens",
    "generated_token_ids",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TokenIdentityParitySummary {
    pub status: TokenIdentityParityStatus,
    pub source_format: &'static str,
    pub steps: usize,
    pub seed_token: TokenId,
    pub vllm_tokens: Vec<TokenId>,
    pub nerva_tokens: Vec<TokenId>,
    pub matched_tokens: usize,
    pub mismatched_tokens: usize,
    pub missing_tokens: usize,
    pub extra_tokens: usize,
    pub first_mismatch_index: Option<usize>,
    pub vllm_token_hash: u64,
    pub nerva_token_hash: u64,
    pub hot_path_allocations: u64,
}

impl TokenIdentityParitySummary {
    pub fn passed(&self) -> bool {
        matches!(self.status, TokenIdentityParityStatus::Ok)
            && self.mismatched_tokens == 0
            && self.missing_tokens == 0
            && self.extra_tokens == 0
            && self.vllm_token_hash == self.nerva_token_hash
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            TokenIdentityParityStatus::Ok => "ok",
            TokenIdentityParityStatus::Mismatch => "mismatch",
        };
        format!(
            "{{\"status\":\"{}\",\"source_format\":\"{}\",\"steps\":{},\"seed_token\":{},\"vllm_tokens\":{},\"nerva_tokens\":{},\"matched_tokens\":{},\"mismatched_tokens\":{},\"missing_tokens\":{},\"extra_tokens\":{},\"first_mismatch_index\":{},\"vllm_token_hash\":{},\"nerva_token_hash\":{},\"hot_path_allocations\":{}}}",
            status,
            json_escape(self.source_format),
            self.steps,
            self.seed_token.0,
            token_ids_to_json(&self.vllm_tokens),
            token_ids_to_json(&self.nerva_tokens),
            self.matched_tokens,
            self.mismatched_tokens,
            self.missing_tokens,
            self.extra_tokens,
            json_opt_usize(self.first_mismatch_index),
            self.vllm_token_hash,
            self.nerva_token_hash,
            self.hot_path_allocations,
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum TokenIdentityParityStatus {
    Ok,
    Mismatch,
}

pub(crate) fn compare_vllm_token_identity(
    vllm_json: &str,
    steps: usize,
) -> Result<TokenIdentityParitySummary, String> {
    let (source_format, vllm_tokens) = parse_vllm_token_ids(vllm_json)?;
    let nerva_summary = nerva_model::tiny_greedy_decode_smoke(steps)
        .map_err(|err| format!("NERVA tiny greedy decode failed: {err:?}"))?;
    let nerva_tokens = nerva_summary.tokens;
    let comparison = compare_token_slices(&vllm_tokens, &nerva_tokens);
    let vllm_token_hash = hash_tokens(&vllm_tokens);
    let nerva_token_hash = hash_tokens(&nerva_tokens);
    let status = if comparison.mismatched_tokens == 0
        && comparison.missing_tokens == 0
        && comparison.extra_tokens == 0
        && vllm_token_hash == nerva_token_hash
        && nerva_summary.hot_path_allocations == 0
    {
        TokenIdentityParityStatus::Ok
    } else {
        TokenIdentityParityStatus::Mismatch
    };

    Ok(TokenIdentityParitySummary {
        status,
        source_format,
        steps,
        seed_token: nerva_summary.seed_token,
        vllm_tokens,
        nerva_tokens,
        matched_tokens: comparison.matched_tokens,
        mismatched_tokens: comparison.mismatched_tokens,
        missing_tokens: comparison.missing_tokens,
        extra_tokens: comparison.extra_tokens,
        first_mismatch_index: comparison.first_mismatch_index,
        vllm_token_hash,
        nerva_token_hash,
        hot_path_allocations: nerva_summary.hot_path_allocations,
    })
}

pub(crate) fn run_vllm_token_identity_parity(
    path: Option<String>,
    steps: usize,
) -> Result<String, String> {
    load_vllm_token_identity_parity(path, steps).map(|summary| summary.to_json())
}

pub(crate) fn load_vllm_token_identity_parity(
    path: Option<String>,
    steps: usize,
) -> Result<TokenIdentityParitySummary, String> {
    let path = path.ok_or_else(|| "vllm-parity requires a vLLM token JSON path".to_string())?;
    let contents =
        fs::read_to_string(&path).map_err(|err| format!("failed to read {path}: {err}"))?;
    compare_vllm_token_identity(&contents, steps)
}

fn parse_vllm_token_ids(source: &str) -> Result<(&'static str, Vec<TokenId>), String> {
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
            return Err(format!("JSON key {key} is missing ':'"));
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

fn compare_token_slices(vllm: &[TokenId], nerva: &[TokenId]) -> TokenComparison {
    let shared = vllm.len().min(nerva.len());
    let mut matched_tokens = 0usize;
    let mut mismatched_tokens = 0usize;
    let mut first_mismatch_index = None;

    for index in 0..shared {
        if vllm[index] == nerva[index] {
            matched_tokens += 1;
        } else {
            mismatched_tokens += 1;
            first_mismatch_index.get_or_insert(index);
        }
    }
    let missing_tokens = nerva.len().saturating_sub(vllm.len());
    let extra_tokens = vllm.len().saturating_sub(nerva.len());
    if first_mismatch_index.is_none() && (missing_tokens > 0 || extra_tokens > 0) {
        first_mismatch_index = Some(shared);
    }

    TokenComparison {
        matched_tokens,
        mismatched_tokens,
        missing_tokens,
        extra_tokens,
        first_mismatch_index,
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct TokenComparison {
    matched_tokens: usize,
    mismatched_tokens: usize,
    missing_tokens: usize,
    extra_tokens: usize,
    first_mismatch_index: Option<usize>,
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

fn hash_tokens(values: &[TokenId]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in values {
        for byte in value.0.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn token_ids_to_json(tokens: &[TokenId]) -> String {
    let mut out = String::from("[");
    for (index, token) in tokens.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&token.0.to_string());
    }
    out.push(']');
    out
}

fn json_opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_vllm_nested_token_ids_for_exact_identity() {
        let summary =
            compare_vllm_token_identity(r#"{"outputs":[{"token_ids":[1,2,3,0]}]}"#, 4).unwrap();

        assert!(summary.passed());
        assert_eq!(summary.source_format, "token_ids");
        assert_eq!(summary.matched_tokens, 4);
        assert_eq!(summary.mismatched_tokens, 0);
        assert_eq!(summary.missing_tokens, 0);
        assert_eq!(summary.extra_tokens, 0);
        assert_eq!(summary.first_mismatch_index, None);
        assert_eq!(summary.vllm_token_hash, summary.nerva_token_hash);
        assert!(summary.to_json().contains("\"status\":\"ok\""));
    }

    #[test]
    fn reports_mismatch_and_first_mismatch_index() {
        let summary = compare_vllm_token_identity(r#"{"output_token_ids":[1,2,99,0]}"#, 4).unwrap();

        assert!(!summary.passed());
        assert_eq!(summary.status, TokenIdentityParityStatus::Mismatch);
        assert_eq!(summary.matched_tokens, 3);
        assert_eq!(summary.mismatched_tokens, 1);
        assert_eq!(summary.first_mismatch_index, Some(2));
        assert_ne!(summary.vllm_token_hash, summary.nerva_token_hash);
    }

    #[test]
    fn reports_missing_and_extra_tokens() {
        let missing = compare_vllm_token_identity(r#"{"tokens":[1,2]}"#, 4).unwrap();
        assert_eq!(missing.missing_tokens, 2);
        assert_eq!(missing.extra_tokens, 0);
        assert_eq!(missing.first_mismatch_index, Some(2));

        let extra =
            compare_vllm_token_identity(r#"{"generated_token_ids":[1,2,3,0,1]}"#, 4).unwrap();
        assert_eq!(extra.missing_tokens, 0);
        assert_eq!(extra.extra_tokens, 1);
        assert_eq!(extra.first_mismatch_index, Some(4));
    }

    #[test]
    fn rejects_missing_or_invalid_token_arrays() {
        assert!(compare_vllm_token_identity(r#"{"text":"hello"}"#, 4).is_err());
        assert!(compare_vllm_token_identity(r#"{"token_ids":[1,-2]}"#, 4).is_err());
        assert!(compare_vllm_token_identity(r#"{"token_ids":["1"]}"#, 4).is_err());
    }
}
