#![forbid(unsafe_code)]

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FieldStatus {
    Match,
    Mismatch,
    MissingBoth,
    MissingBaseline,
    MissingCandidate,
}

impl FieldStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Mismatch => "mismatch",
            Self::MissingBoth => "missing_both",
            Self::MissingBaseline => "missing_baseline",
            Self::MissingCandidate => "missing_candidate",
        }
    }

    pub const fn is_mismatch(self) -> bool {
        matches!(
            self,
            Self::Mismatch | Self::MissingBaseline | Self::MissingCandidate
        )
    }

    pub const fn is_missing(self) -> bool {
        matches!(
            self,
            Self::MissingBoth | Self::MissingBaseline | Self::MissingCandidate
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FieldComparison {
    pub name: &'static str,
    pub baseline: Option<i128>,
    pub candidate: Option<i128>,
    pub delta: Option<i128>,
    pub tolerance: i128,
    pub status: FieldStatus,
}

impl FieldComparison {
    pub fn to_json(self) -> String {
        format!(
            "{{\"field\":\"{}\",\"baseline\":{},\"candidate\":{},\"delta\":{},\"tolerance\":{},\"status\":\"{}\"}}",
            json_escape(self.name),
            json_opt_i128(self.baseline),
            json_opt_i128(self.candidate),
            json_opt_i128(self.delta),
            self.tolerance,
            self.status.as_str(),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerComparison {
    pub fields: Vec<FieldComparison>,
    pub mismatch_count: usize,
    pub missing_count: usize,
}

impl LedgerComparison {
    pub fn status(&self) -> &'static str {
        if self.mismatch_count == 0 {
            "ok"
        } else {
            "mismatch"
        }
    }

    pub fn to_json(&self) -> String {
        let fields = self
            .fields
            .iter()
            .map(|field| field.to_json())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"status\":\"{}\",\"compared_fields\":{},\"mismatches\":{},\"missing_fields\":{},\"fields\":[{}]}}",
            self.status(),
            self.fields.len(),
            self.mismatch_count,
            self.missing_count,
            fields,
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct FieldSpec {
    name: &'static str,
    aliases: &'static [&'static str],
    tolerance_field: bool,
}

const FIELD_SPECS: &[FieldSpec] = &[
    FieldSpec {
        name: "total_latency_ns",
        aliases: &["total_latency_ns", "wall_latency_ns"],
        tolerance_field: true,
    },
    FieldSpec {
        name: "host_sync_events",
        aliases: &["host_wait_events", "sync_events", "sync_calls"],
        tolerance_field: false,
    },
    FieldSpec {
        name: "hot_path_allocations",
        aliases: &["hot_path_allocations", "allocator_calls"],
        tolerance_field: false,
    },
    FieldSpec {
        name: "h2d_bytes",
        aliases: &["H2D_bytes", "h2d_bytes"],
        tolerance_field: false,
    },
    FieldSpec {
        name: "d2h_bytes",
        aliases: &["D2H_bytes", "d2h_bytes"],
        tolerance_field: false,
    },
    FieldSpec {
        name: "kv_residency_decisions",
        aliases: &["residency_decisions", "decisions"],
        tolerance_field: false,
    },
    FieldSpec {
        name: "kernel_events",
        aliases: &["kernel_events", "kernel_count"],
        tolerance_field: false,
    },
    FieldSpec {
        name: "graph_replay_events",
        aliases: &["graph_replay_events", "graph_launches"],
        tolerance_field: false,
    },
    FieldSpec {
        name: "gpu_idle_ns",
        aliases: &["device_timeline_idle_ns", "gpu_idle_ns"],
        tolerance_field: true,
    },
];

pub fn compare_ledgers(
    baseline_json: &str,
    candidate_json: &str,
    latency_tolerance_ns: i128,
) -> LedgerComparison {
    let mut fields = Vec::with_capacity(FIELD_SPECS.len());
    let mut mismatch_count = 0usize;
    let mut missing_count = 0usize;

    for spec in FIELD_SPECS {
        let baseline = find_first_number_by_aliases(baseline_json, spec.aliases);
        let candidate = find_first_number_by_aliases(candidate_json, spec.aliases);
        let tolerance = if spec.tolerance_field {
            latency_tolerance_ns.max(0)
        } else {
            0
        };
        let delta = match (baseline, candidate) {
            (Some(lhs), Some(rhs)) => Some(rhs - lhs),
            _ => None,
        };
        let status = match (baseline, candidate, delta) {
            (None, None, _) => FieldStatus::MissingBoth,
            (None, Some(_), _) => FieldStatus::MissingBaseline,
            (Some(_), None, _) => FieldStatus::MissingCandidate,
            (Some(_), Some(_), Some(delta)) if delta.abs() <= tolerance => FieldStatus::Match,
            (Some(_), Some(_), Some(_)) => FieldStatus::Mismatch,
            (Some(_), Some(_), None) => FieldStatus::Mismatch,
        };
        if status.is_mismatch() {
            mismatch_count += 1;
        }
        if status.is_missing() {
            missing_count += 1;
        }
        fields.push(FieldComparison {
            name: spec.name,
            baseline,
            candidate,
            delta,
            tolerance,
            status,
        });
    }

    LedgerComparison {
        fields,
        mismatch_count,
        missing_count,
    }
}

fn find_first_number_by_aliases(source: &str, aliases: &[&str]) -> Option<i128> {
    aliases
        .iter()
        .find_map(|alias| find_first_number(source, alias))
}

fn find_first_number(source: &str, key: &str) -> Option<i128> {
    let bytes = source.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] != b'"' {
            index += 1;
            continue;
        }
        let (field, after_string) = parse_json_string(source, index)?;
        index = after_string;
        if field != key {
            continue;
        }
        let colon = skip_ws(bytes, after_string);
        if bytes.get(colon) != Some(&b':') {
            continue;
        }
        let value_start = skip_ws(bytes, colon + 1);
        if let Some(number) = parse_json_integer(source, value_start) {
            return Some(number);
        }
    }
    None
}

fn parse_json_integer(source: &str, start: usize) -> Option<i128> {
    let bytes = source.as_bytes();
    let mut end = start;
    if bytes.get(end) == Some(&b'-') {
        end += 1;
    }
    let digit_start = end;
    while matches!(bytes.get(end), Some(b'0'..=b'9')) {
        end += 1;
    }
    if end == digit_start {
        return None;
    }
    source[start..end].parse::<i128>().ok()
}

fn parse_json_string(source: &str, start: usize) -> Option<(String, usize)> {
    let bytes = source.as_bytes();
    if bytes.get(start) != Some(&b'"') {
        return None;
    }
    let mut index = start + 1;
    let mut value = String::new();
    while index < bytes.len() {
        match bytes[index] {
            b'"' => return Some((value, index + 1)),
            b'\\' => {
                index += 1;
                match bytes.get(index).copied()? {
                    b'"' => value.push('"'),
                    b'\\' => value.push('\\'),
                    b'/' => value.push('/'),
                    b'b' => value.push('\u{0008}'),
                    b'f' => value.push('\u{000c}'),
                    b'n' => value.push('\n'),
                    b'r' => value.push('\r'),
                    b't' => value.push('\t'),
                    _ => return None,
                }
            }
            byte => value.push(byte as char),
        }
        index += 1;
    }
    None
}

fn skip_ws(bytes: &[u8], mut index: usize) -> usize {
    while matches!(bytes.get(index), Some(b' ' | b'\n' | b'\r' | b'\t')) {
        index += 1;
    }
    index
}

pub fn json_escape(value: &str) -> String {
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

fn json_opt_i128(value: Option<i128>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::{FieldStatus, compare_ledgers, json_escape};

    #[test]
    fn comparison_reports_matching_core_summary_fields() {
        let baseline = r#"{
            "summary": {
                "total_latency_ns": 100,
                "host_wait_events": 4,
                "hot_path_allocations": 0,
                "kernel_events": 8,
                "graph_replay_events": 4,
                "device_timeline_idle_ns": 0
            }
        }"#;
        let candidate = r#"{
            "total_latency_ns": 105,
            "host_wait_events": 4,
            "hot_path_allocations": 0,
            "kernel_events": 8,
            "graph_replay_events": 4,
            "device_timeline_idle_ns": 0
        }"#;

        let comparison = compare_ledgers(baseline, candidate, 5);

        assert_eq!(comparison.status(), "ok");
        assert_eq!(comparison.mismatch_count, 0);
        assert!(
            comparison
                .fields
                .iter()
                .any(|field| field.name == "total_latency_ns"
                    && field.delta == Some(5)
                    && field.status == FieldStatus::Match)
        );
    }

    #[test]
    fn comparison_flags_count_mismatches_without_latency_tolerance() {
        let baseline = r#"{"sync_events":7,"hot_path_allocations":0}"#;
        let candidate = r#"{"sync_events":8,"hot_path_allocations":1}"#;

        let comparison = compare_ledgers(baseline, candidate, 1000);

        assert_eq!(comparison.status(), "mismatch");
        assert_eq!(comparison.mismatch_count, 2);
        assert!(
            comparison
                .fields
                .iter()
                .any(|field| field.name == "host_sync_events"
                    && field.delta == Some(1)
                    && field.status == FieldStatus::Mismatch)
        );
    }

    #[test]
    fn comparison_treats_one_sided_missing_fields_as_mismatches() {
        let comparison = compare_ledgers(r#"{"H2D_bytes":128}"#, r#"{}"#, 0);

        assert_eq!(comparison.status(), "mismatch");
        assert!(comparison.fields.iter().any(
            |field| field.name == "h2d_bytes" && field.status == FieldStatus::MissingCandidate
        ));
    }

    #[test]
    fn comparison_json_escapes_fields_and_reports_missing_both() {
        let comparison = compare_ledgers("{}", "{}", 0);
        let json = comparison.to_json();

        assert_eq!(comparison.status(), "ok");
        assert!(json.contains("\"missing_fields\":9"));
        assert!(json.contains("\"status\":\"missing_both\""));
        assert_eq!(json_escape("quote\" line\n"), "quote\\\" line\\n");
    }
}
