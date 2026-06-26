use nerva_ledger::types::metric::MetricSource;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MeasurementKind {
    CpuCopy,
    CpuKernel,
    Merge,
    Queue,
    Sync,
}

impl MeasurementKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CpuCopy => "cpu_copy",
            Self::CpuKernel => "cpu_kernel",
            Self::Merge => "merge",
            Self::Queue => "queue",
            Self::Sync => "sync",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MeasurementEntry {
    pub kind: MeasurementKind,
    pub label: &'static str,
    pub bytes: usize,
    pub iterations: u64,
    pub elapsed_ns: u64,
    pub effective_bandwidth_bps: u64,
    pub source: MetricSource,
}

impl MeasurementEntry {
    pub fn runtime_timestamp(
        kind: MeasurementKind,
        label: &'static str,
        bytes: usize,
        iterations: u64,
        elapsed_ns: u64,
    ) -> Self {
        let total_bytes = (bytes as u128).saturating_mul(iterations as u128);
        let effective_bandwidth_bps = if elapsed_ns == 0 {
            0
        } else {
            (total_bytes.saturating_mul(1_000_000_000) / elapsed_ns as u128).min(u64::MAX as u128)
                as u64
        };
        Self {
            kind,
            label,
            bytes,
            iterations,
            elapsed_ns,
            effective_bandwidth_bps,
            source: MetricSource::RuntimeTimestamp,
        }
    }
}
