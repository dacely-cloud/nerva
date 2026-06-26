use nerva_runtime::engine::runtime::Runtime;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_security_isolation(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_security_isolation_probe() {
        Ok(summary) => report.push(
            "security_isolation",
            summary.passed()
                && summary.sensitive_blocks == 3
                && summary.bytes_sanitized > 0
                && summary.zero_fill_events == 3
                && summary.version_revocations == 3
                && summary.hot_path_sanitize_rejections == 1
                && summary.non_sensitive_rejections == 1
                && summary.unready_rejections == 1
                && summary.stale_version_rejections == 3
                && summary.ready_after_sanitize == 3
                && summary.owner_cleared_after_sanitize == 3
                && summary.hot_path_allocations == 0,
            format!(
                "sensitive_blocks={} bytes_sanitized={} zero_fill_events={} version_revocations={} hot_path_sanitize_rejections={} non_sensitive_rejections={} unready_rejections={} stale_version_rejections={} ready_after_sanitize={} owner_cleared_after_sanitize={} hot_path_allocations={}",
                summary.sensitive_blocks,
                summary.bytes_sanitized,
                summary.zero_fill_events,
                summary.version_revocations,
                summary.hot_path_sanitize_rejections,
                summary.non_sensitive_rejections,
                summary.unready_rejections,
                summary.stale_version_rejections,
                summary.ready_after_sanitize,
                summary.owner_cleared_after_sanitize,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("security_isolation", false, format!("{err:?}")),
    }
}
