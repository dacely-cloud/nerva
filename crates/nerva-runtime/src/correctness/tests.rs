use crate::correctness::case::CorrectnessCase;
use crate::correctness::exactness::ExactnessClass;
use crate::correctness::probe::run_correctness_validation_probe;
use crate::correctness::validator::validate_correctness_case;

#[test]
fn correctness_validation_probe_reports_core_exactness_classes() {
    let summary = run_correctness_validation_probe().unwrap();

    assert!(summary.passed());
    assert_eq!(summary.accepted_cases, 3);
    assert_eq!(summary.bit_exact_cases, 1);
    assert_eq!(summary.fp_tolerance_cases, 1);
    assert_eq!(summary.distribution_preserving_cases, 1);
    assert_eq!(summary.approximate_rejections, 1);
    assert_eq!(summary.hot_path_allocations, 0);
}

#[test]
fn correctness_validator_rejects_approximate_core_claims() {
    let result = validate_correctness_case(CorrectnessCase {
        name: "approximate_rejection",
        exactness: ExactnessClass::Approximate,
        expected_hash: 7,
        observed_hash: 7,
        max_abs_error_micros: 0,
        tolerance_micros: 0,
    });

    assert!(result.is_err());
}
