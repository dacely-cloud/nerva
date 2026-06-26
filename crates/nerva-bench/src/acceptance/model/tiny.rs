use crate::acceptance::cuda;
use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_tiny_model_and_cuda_decode(report: &mut AcceptanceReport) {
    match nerva_model::tiny::smoke::tiny_greedy_decode_smoke(8) {
        Ok(summary) => {
            report.push(
                "tiny_model_greedy_parity",
                summary.parity
                    && summary.ledger_count == summary.steps as u64
                    && summary.hot_path_allocations == 0,
                format!(
                    "steps={} parity={} ledger_count={} device_events={} hot_path_allocations={} output_hash={}",
                    summary.steps,
                    summary.parity,
                    summary.ledger_count,
                    summary.device_events,
                    summary.hot_path_allocations,
                    summary.output_hash,
                ),
            );
            cuda::decode::push_tiny_decode_check(report, &summary);
        }
        Err(err) => {
            let details = format!("{err:?}");
            report.push("tiny_model_greedy_parity", false, details.clone());
            cuda::decode::push_prerequisite_failure(report, &details);
        }
    }
}
