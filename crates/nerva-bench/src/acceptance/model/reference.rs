use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_reference_block(report: &mut AcceptanceReport) {
    match nerva_model::reference::smoke::run::reference_block_smoke() {
        Ok(summary) => report.push(
            "reference_block",
            summary.hot_path_allocations == 0,
            format!(
                "hidden={} heads={} output_hash={} hot_path_allocations={}",
                summary.hidden, summary.heads, summary.output_hash, summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("reference_block", false, format!("{err:?}")),
    }
}
