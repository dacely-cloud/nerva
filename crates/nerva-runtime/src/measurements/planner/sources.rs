use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::DeviceOrdinal;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::metric::MetricSource;

use crate::measurements::entry::{MeasurementEntry, MeasurementKind};
use crate::measurements::planner::candidate::MeasuredPlannerCandidate;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PlannerMeasurementSources {
    pub source_measurements: u64,
    pub runtime_timestamp_entries: u64,
    cpu_kernel_ns: u64,
    cpu_copy_ns: u64,
    merge_ns: u64,
    queue_ns: u64,
    sync_ns: u64,
}

impl PlannerMeasurementSources {
    pub(crate) fn from_entries(entries: &[MeasurementEntry]) -> Result<Self> {
        Ok(Self {
            source_measurements: entries.len() as u64,
            runtime_timestamp_entries: entries
                .iter()
                .filter(|entry| entry.source == MetricSource::RuntimeTimestamp)
                .count() as u64,
            cpu_kernel_ns: per_iteration_ns(required_entry(entries, MeasurementKind::CpuKernel)?),
            cpu_copy_ns: per_iteration_ns(required_entry(entries, MeasurementKind::CpuCopy)?),
            merge_ns: per_iteration_ns(required_entry(entries, MeasurementKind::Merge)?),
            queue_ns: per_iteration_ns(required_entry(entries, MeasurementKind::Queue)?),
            sync_ns: per_iteration_ns(required_entry(entries, MeasurementKind::Sync)?),
        })
    }

    pub(crate) fn candidates(&self) -> Vec<MeasuredPlannerCandidate> {
        vec![
            MeasuredPlannerCandidate {
                label: "cpu_dram_exact_matvec",
                executor: ExecutionOwner::Cpu,
                visible_ns: self
                    .cpu_kernel_ns
                    .saturating_add(self.queue_ns)
                    .saturating_add(self.sync_ns),
            },
            MeasuredPlannerCandidate {
                label: "gpu_staged_exact_matvec",
                executor: ExecutionOwner::Gpu(DeviceOrdinal(0)),
                visible_ns: self
                    .cpu_copy_ns
                    .saturating_add(self.queue_ns)
                    .saturating_add(self.sync_ns),
            },
            MeasuredPlannerCandidate {
                label: "split_cpu_gpu_exact_matvec",
                executor: ExecutionOwner::PhaseTransition,
                visible_ns: self
                    .cpu_kernel_ns
                    .saturating_div(2)
                    .saturating_add(self.cpu_copy_ns.saturating_div(2))
                    .saturating_add(self.merge_ns)
                    .saturating_add(self.queue_ns)
                    .saturating_add(self.sync_ns),
            },
        ]
    }
}

fn required_entry(
    entries: &[MeasurementEntry],
    kind: MeasurementKind,
) -> Result<&MeasurementEntry> {
    entries
        .iter()
        .find(|entry| entry.kind == kind)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("measurement table missing {}", kind.as_str()),
        })
}

fn per_iteration_ns(entry: &MeasurementEntry) -> u64 {
    if entry.iterations == 0 {
        entry.elapsed_ns.max(1)
    } else {
        entry.elapsed_ns.saturating_div(entry.iterations).max(1)
    }
}
