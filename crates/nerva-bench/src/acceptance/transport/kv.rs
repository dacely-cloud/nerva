use nerva_runtime::engine::kv_attention::config::TieredKvAttentionProbeConfig;
use nerva_runtime::engine::kv_attention::summary::TieredKvAttentionProbeStatus;
use nerva_runtime::engine::kv_probe::config::KvResidencyProbeConfig;
use nerva_runtime::engine::kv_probe::summary::KvResidencyProbeStatus;
use nerva_runtime::engine::runtime::Runtime;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_kv_residency(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_kv_residency_probe(KvResidencyProbeConfig::default()) {
        Ok(summary) => report.push(
            "kv_residency_tiering",
            matches!(summary.status, KvResidencyProbeStatus::Ok)
                && summary.decisions > 0
                && summary.prefetches > 0
                && summary.demotions > 0
                && summary.evictions > 0
                && summary.stall_events > 0
                && summary.hot_path_allocations == 0,
            format!(
                "pages={} decisions={} prefetches={} demotions={} evictions={} stall_events={} hot_path_allocations={}",
                summary.pages,
                summary.decisions,
                summary.prefetches,
                summary.demotions,
                summary.evictions,
                summary.stall_events,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("kv_residency_tiering", false, format!("{err:?}")),
    }
}

pub(crate) fn push_tiered_kv_attention(report: &mut AcceptanceReport, runtime: &Runtime) {
    match runtime.run_tiered_kv_attention_probe(TieredKvAttentionProbeConfig::default()) {
        Ok(summary) => report.push(
            "tiered_kv_attention_execution",
            matches!(summary.status, TieredKvAttentionProbeStatus::Ok)
                && summary.parity
                && summary.pages == 2
                && summary.tokens == 4
                && summary.dram_pages == 1
                && summary.vram_pages == 1
                && summary.cpu_block_events == 1
                && summary.device_block_events == 1
                && summary.execution_decisions == 2
                && summary.block_version_dependencies == 2
                && summary.hot_path_allocations == 0,
            format!(
                "pages={} tokens={} dram_pages={} vram_pages={} parity={} max_abs_error={} execution_decisions={} block_version_dependencies={} cpu_block_events={} device_block_events={} hot_path_allocations={}",
                summary.pages,
                summary.tokens,
                summary.dram_pages,
                summary.vram_pages,
                summary.parity,
                summary.max_abs_error,
                summary.execution_decisions,
                summary.block_version_dependencies,
                summary.cpu_block_events,
                summary.device_block_events,
                summary.hot_path_allocations,
            ),
        ),
        Err(err) => report.push("tiered_kv_attention_execution", false, format!("{err:?}")),
    }
}
