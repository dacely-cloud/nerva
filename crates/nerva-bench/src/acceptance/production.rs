use nerva_runtime::engine::runtime::Runtime;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_production_invariants(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_production_invariant_probe() {
        Ok(summary) => report.push(
            "production_ledger_invariants",
            summary.passed()
                && summary.accepted_ledgers == 1
                && summary.classified_sync_ledgers == 1
                && summary.measured_fallbacks == 2
                && summary.debug_sync_rejections == 1
                && summary.debug_fallback_rejections == 1
                && summary.unmeasured_fallback_rejections == 1
                && summary.unnamed_fallback_rejections == 1
                && summary.hot_path_allocations == 0,
            format!(
                "accepted_ledgers={} classified_sync_ledgers={} measured_fallbacks={} debug_sync_rejections={} debug_fallback_rejections={} unmeasured_fallback_rejections={} unnamed_fallback_rejections={} hot_path_allocations={}",
                summary.accepted_ledgers,
                summary.classified_sync_ledgers,
                summary.measured_fallbacks,
                summary.debug_sync_rejections,
                summary.debug_fallback_rejections,
                summary.unmeasured_fallback_rejections,
                summary.unnamed_fallback_rejections,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("production_ledger_invariants", false, format!("{err:?}")),
    }
}
