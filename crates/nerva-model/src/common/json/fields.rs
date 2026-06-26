use nerva_core::types::error::{NervaError, Result};

use crate::common::json::parse::{
    find_top_level_json_value, parse_json_string_at, parse_json_string_value, skip_json_ws,
};

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
