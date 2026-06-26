use crate::acceptance::cuda;
use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_tiered_attention_and_cuda(report: &mut AcceptanceReport) {
    match nerva_model::attention::smoke::blockwise_attention_smoke() {
        Ok(summary) => {
            report.push(
                "tiered_blockwise_attention",
                summary.cpu_block_events > 0
                    && summary.device_block_events > 0
                    && summary.hot_path_allocations == 0,
                format!(
                    "blocks={} tokens={} cpu_block_events={} device_block_events={} hot_path_allocations={} output_hash={}",
                    summary.blocks,
                    summary.tokens,
                    summary.cpu_block_events,
                    summary.device_block_events,
                    summary.hot_path_allocations,
                    summary.output_hash,
                ),
            );
            cuda::attention::push_tiered_check(report, &summary);
        }
        Err(err) => {
            let details = format!("{err:?}");
            report.push("tiered_blockwise_attention", false, details.clone());
            cuda::attention::push_prerequisite_failure(report, &details);
        }
    }
}
