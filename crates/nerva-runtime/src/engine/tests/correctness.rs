use crate::engine::runtime::{Runtime, RuntimeConfig};

#[test]
fn correctness_validation_probe_rejects_approximate_core_results() {
    let runtime = Runtime::new(RuntimeConfig::default()).unwrap();
    let summary = runtime.run_correctness_validation_probe().unwrap();

    assert!(summary.passed());
    assert_eq!(summary.accepted_cases, 3);
    assert_eq!(summary.approximate_rejections, 1);
    assert_eq!(summary.bit_exact_mismatch_rejections, 1);
    assert_eq!(summary.tolerance_rejections, 1);
    assert_eq!(summary.hot_path_allocations, 0);
}
