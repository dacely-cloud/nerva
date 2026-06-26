use nerva_runtime::engine::runtime::Runtime;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_correctness_validation(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_correctness_validation_probe() {
        Ok(summary) => report.push(
            "correctness_exactness_validation",
            summary.passed()
                && summary.accepted_cases == 3
                && summary.bit_exact_cases == 1
                && summary.fp_tolerance_cases == 1
                && summary.distribution_preserving_cases == 1
                && summary.approximate_rejections == 1
                && summary.bit_exact_mismatch_rejections == 1
                && summary.tolerance_rejections == 1
                && summary.exactness_classes_declared == 3
                && summary.hot_path_allocations == 0,
            format!(
                "accepted_cases={} bit_exact_cases={} fp_tolerance_cases={} distribution_preserving_cases={} approximate_rejections={} bit_exact_mismatch_rejections={} tolerance_rejections={} exactness_classes_declared={} hot_path_allocations={}",
                summary.accepted_cases,
                summary.bit_exact_cases,
                summary.fp_tolerance_cases,
                summary.distribution_preserving_cases,
                summary.approximate_rejections,
                summary.bit_exact_mismatch_rejections,
                summary.tolerance_rejections,
                summary.exactness_classes_declared,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push(
            "correctness_exactness_validation",
            false,
            format!("{err:?}"),
        ),
    }
}
