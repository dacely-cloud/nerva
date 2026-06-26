use crate::security::probe::run_security_isolation_probe;

#[test]
fn security_isolation_probe_sanitizes_and_revokes_sensitive_blocks() {
    let summary = run_security_isolation_probe().unwrap();

    assert!(summary.passed());
    assert_eq!(summary.sensitive_blocks, 3);
    assert_eq!(summary.bytes_sanitized, 16_384);
    assert_eq!(summary.zero_fill_events, 3);
    assert_eq!(summary.version_revocations, 3);
    assert_eq!(summary.hot_path_sanitize_rejections, 1);
    assert_eq!(summary.non_sensitive_rejections, 1);
    assert_eq!(summary.unready_rejections, 1);
    assert_eq!(summary.stale_version_rejections, 3);
    assert_eq!(summary.ready_after_sanitize, 3);
    assert_eq!(summary.owner_cleared_after_sanitize, 3);
    assert_eq!(summary.hot_path_allocations, 0);
    assert!(summary.to_json().contains("\"status\":\"ok\""));
    assert!(summary.to_json().contains("\"version_revocations\":3"));
}
