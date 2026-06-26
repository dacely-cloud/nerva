#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SecurityIsolationStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SecurityIsolationSummary {
    pub status: SecurityIsolationStatus,
    pub sensitive_blocks: u64,
    pub bytes_sanitized: usize,
    pub zero_fill_events: u64,
    pub version_revocations: u64,
    pub hot_path_sanitize_rejections: u64,
    pub non_sensitive_rejections: u64,
    pub unready_rejections: u64,
    pub stale_version_rejections: u64,
    pub ready_after_sanitize: u64,
    pub owner_cleared_after_sanitize: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl SecurityIsolationSummary {
    pub fn passed(self) -> bool {
        matches!(self.status, SecurityIsolationStatus::Ok)
            && self.sensitive_blocks == 3
            && self.bytes_sanitized > 0
            && self.zero_fill_events == 3
            && self.version_revocations == 3
            && self.hot_path_sanitize_rejections == 1
            && self.non_sensitive_rejections == 1
            && self.unready_rejections == 1
            && self.stale_version_rejections == 3
            && self.ready_after_sanitize == 3
            && self.owner_cleared_after_sanitize == 3
            && self.hot_path_allocations == 0
    }

    pub fn to_json(self) -> String {
        let status = match self.status {
            SecurityIsolationStatus::Ok => "ok",
            SecurityIsolationStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"sensitive_blocks\":{},\"bytes_sanitized\":{},\"zero_fill_events\":{},\"version_revocations\":{},\"hot_path_sanitize_rejections\":{},\"non_sensitive_rejections\":{},\"unready_rejections\":{},\"stale_version_rejections\":{},\"ready_after_sanitize\":{},\"owner_cleared_after_sanitize\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.sensitive_blocks,
            self.bytes_sanitized,
            self.zero_fill_events,
            self.version_revocations,
            self.hot_path_sanitize_rejections,
            self.non_sensitive_rejections,
            self.unready_rejections,
            self.stale_version_rejections,
            self.ready_after_sanitize,
            self.owner_cleared_after_sanitize,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}

fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{value}\""))
}
