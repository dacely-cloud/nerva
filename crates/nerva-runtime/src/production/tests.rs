use crate::production::probe::run_production_invariant_probe;

#[test]
fn production_invariant_probe_rejects_debug_and_unmeasured_paths() {
    let summary = run_production_invariant_probe().unwrap();

    assert!(summary.passed());
    assert_eq!(summary.accepted_ledgers, 1);
    assert_eq!(summary.measured_fallbacks, 2);
    assert_eq!(summary.debug_sync_rejections, 1);
    assert_eq!(summary.debug_fallback_rejections, 1);
    assert_eq!(summary.unmeasured_fallback_rejections, 1);
    assert_eq!(summary.unnamed_fallback_rejections, 1);
    assert_eq!(summary.hot_path_allocations, 0);
}
