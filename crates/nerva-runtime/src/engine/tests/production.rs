use crate::engine::runtime::{Runtime, RuntimeConfig};

#[test]
fn runtime_production_invariant_probe_rejects_debug_paths() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_production_invariant_probe().unwrap();

    assert!(summary.passed());
    assert_eq!(summary.debug_sync_rejections, 1);
    assert_eq!(summary.debug_fallback_rejections, 1);
    assert_eq!(summary.unmeasured_fallback_rejections, 1);
}
