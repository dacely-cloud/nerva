use crate::engine::runtime::{Runtime, RuntimeConfig};

#[test]
fn security_isolation_probe_revokes_sensitive_buffer_versions() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_security_isolation_probe().unwrap();

    assert!(summary.passed());
    assert_eq!(summary.sensitive_blocks, 3);
    assert_eq!(summary.zero_fill_events, 3);
    assert_eq!(summary.version_revocations, 3);
    assert_eq!(summary.hot_path_sanitize_rejections, 1);
    assert_eq!(summary.hot_path_allocations, 0);
}
