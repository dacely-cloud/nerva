use nerva_ledger::types::metric::MetricSource;

use crate::measurements::entry::{MeasurementEntry, MeasurementKind};
use crate::measurements::table::MeasurementTable;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MeasurementTableStatus {
    Ok,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasurementTableSummary {
    pub status: MeasurementTableStatus,
    pub entries: Vec<MeasurementEntry>,
    pub measured_entries: u64,
    pub estimated_entries: u64,
    pub runtime_timestamp_entries: u64,
    pub cpu_copy_entries: u64,
    pub cpu_kernel_entries: u64,
    pub merge_entries: u64,
    pub queue_entries: u64,
    pub sync_entries: u64,
    pub total_latency_ns: u64,
    pub min_effective_bandwidth_bps: u64,
    pub all_nonzero_latency: bool,
    pub all_measured: bool,
    pub hot_path_allocations: u64,
}

impl MeasurementTableSummary {
    pub fn from_table(table: MeasurementTable, hot_path_allocations: u64) -> Self {
        Self {
            measured_entries: table.len() as u64,
            estimated_entries: table.count_source(MetricSource::EstimatedModel),
            runtime_timestamp_entries: table.count_source(MetricSource::RuntimeTimestamp),
            cpu_copy_entries: table.count_kind(MeasurementKind::CpuCopy),
            cpu_kernel_entries: table.count_kind(MeasurementKind::CpuKernel),
            merge_entries: table.count_kind(MeasurementKind::Merge),
            queue_entries: table.count_kind(MeasurementKind::Queue),
            sync_entries: table.count_kind(MeasurementKind::Sync),
            total_latency_ns: table.total_latency_ns(),
            min_effective_bandwidth_bps: table.min_bandwidth_bps(),
            all_nonzero_latency: table.all_nonzero_latency(),
            all_measured: table.all_measured(),
            status: MeasurementTableStatus::Ok,
            entries: table.entries,
            hot_path_allocations,
        }
    }

    pub fn passed(&self) -> bool {
        matches!(self.status, MeasurementTableStatus::Ok)
            && self.measured_entries >= 5
            && self.estimated_entries == 0
            && self.runtime_timestamp_entries == self.measured_entries
            && self.cpu_copy_entries > 0
            && self.cpu_kernel_entries > 0
            && self.merge_entries > 0
            && self.queue_entries > 0
            && self.sync_entries > 0
            && self.total_latency_ns > 0
            && self.min_effective_bandwidth_bps > 0
            && self.all_nonzero_latency
            && self.all_measured
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            MeasurementTableStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"measured_entries\":{},\"estimated_entries\":{},\"runtime_timestamp_entries\":{},\"cpu_copy_entries\":{},\"cpu_kernel_entries\":{},\"merge_entries\":{},\"queue_entries\":{},\"sync_entries\":{},\"total_latency_ns\":{},\"min_effective_bandwidth_bps\":{},\"all_nonzero_latency\":{},\"all_measured\":{},\"hot_path_allocations\":{},\"entries\":{}}}",
            status,
            self.measured_entries,
            self.estimated_entries,
            self.runtime_timestamp_entries,
            self.cpu_copy_entries,
            self.cpu_kernel_entries,
            self.merge_entries,
            self.queue_entries,
            self.sync_entries,
            self.total_latency_ns,
            self.min_effective_bandwidth_bps,
            self.all_nonzero_latency,
            self.all_measured,
            self.hot_path_allocations,
            entries_json(&self.entries),
        )
    }
}

fn entries_json(entries: &[MeasurementEntry]) -> String {
    let items = entries.iter().map(entry_json).collect::<Vec<_>>().join(",");
    format!("[{items}]")
}

fn entry_json(entry: &MeasurementEntry) -> String {
    format!(
        "{{\"kind\":\"{}\",\"label\":\"{}\",\"bytes\":{},\"iterations\":{},\"elapsed_ns\":{},\"effective_bandwidth_bps\":{},\"metric_source\":\"{}\"}}",
        entry.kind.as_str(),
        json_escape(entry.label),
        entry.bytes,
        entry.iterations,
        entry.elapsed_ns,
        entry.effective_bandwidth_bps,
        entry.source.as_str(),
    )
}

fn json_escape(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out
}
