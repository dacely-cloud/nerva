pub(crate) fn parse_optional_u64(
    value: Option<String>,
    default: u64,
    label: &'static str,
) -> Result<u64, String> {
    match value {
        Some(value) => value
            .parse::<u64>()
            .map_err(|_| format!("{label} must be an unsigned integer")),
        None => Ok(default),
    }
}

pub(crate) fn parse_optional_u32(
    value: Option<String>,
    default: u32,
    label: &'static str,
) -> Result<u32, String> {
    match value {
        Some(value) => value
            .parse::<u32>()
            .map_err(|_| format!("{label} must be an unsigned 32-bit integer")),
        None => Ok(default),
    }
}

pub(crate) fn parse_optional_usize(
    value: Option<String>,
    default: usize,
    label: &'static str,
) -> Result<usize, String> {
    match value {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{label} must be an unsigned integer")),
        None => Ok(default),
    }
}
