use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_warm_compute(report: &mut AcceptanceReport) {
    match nerva_model::warm_compute::probe::run::warm_compute_probe() {
        Ok(summary) => report.push(
            "warm_compute_selection",
            summary.parity
                && summary.cpu_beats_staged
                && summary.execution_decisions > 0
                && summary.runtime_timestamp_decisions > 0
                && summary.measured_candidate_costs >= summary.candidates.len() as u64
                && summary.estimated_candidate_costs >= summary.candidates.len() as u64
                && summary.hot_path_allocations == 0,
            format!(
                "selected_strategy={} parity={} cpu_beats_staged={} execution_decisions={} runtime_timestamp_decisions={} measured_candidate_costs={} estimated_candidate_costs={} hot_path_allocations={} output_hash={}",
                summary.selected_strategy.label(),
                summary.parity,
                summary.cpu_beats_staged,
                summary.execution_decisions,
                summary.runtime_timestamp_decisions,
                summary.measured_candidate_costs,
                summary.estimated_candidate_costs,
                summary.hot_path_allocations,
                summary.output_hash,
            ),
        ),
        Err(err) => report.push("warm_compute_selection", false, format!("{err:?}")),
    }
}
