use nerva_ledger::types::metric::MetricSource;

use crate::measurements::entry::{MeasurementEntry, MeasurementKind};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasurementTable {
    pub entries: Vec<MeasurementEntry>,
}

impl MeasurementTable {
    pub fn new(entries: Vec<MeasurementEntry>) -> Self {
        Self { entries }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn count_kind(&self, kind: MeasurementKind) -> u64 {
        self.entries
            .iter()
            .filter(|entry| entry.kind == kind)
            .count() as u64
    }

    pub fn count_source(&self, source: MetricSource) -> u64 {
        self.entries
            .iter()
            .filter(|entry| entry.source == source)
            .count() as u64
    }

    pub fn total_latency_ns(&self) -> u64 {
        self.entries.iter().map(|entry| entry.elapsed_ns).sum()
    }

    pub fn min_bandwidth_bps(&self) -> u64 {
        self.entries
            .iter()
            .map(|entry| entry.effective_bandwidth_bps)
            .min()
            .unwrap_or(0)
    }

    pub fn all_nonzero_latency(&self) -> bool {
        self.entries.iter().all(|entry| entry.elapsed_ns > 0)
    }

    pub fn all_measured(&self) -> bool {
        self.entries
            .iter()
            .all(|entry| entry.source != MetricSource::EstimatedModel)
    }
}
