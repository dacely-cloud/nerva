use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::device::DeviceOrdinal;

use nerva_core::types::memory::tier::MemoryTier;

use crate::types::metric::MetricSource;
use crate::types::sync::SyncClass;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LedgerEventKind {
    GraphReplay,
    KernelLaunch,
    CpuActivity,
    DeviceActivity,
    Copy,
    Sync,
    Allocation,
    Eviction,
    Prefetch,
    Stall,
    Transport,
}

impl LedgerEventKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GraphReplay => "graph_replay",
            Self::KernelLaunch => "kernel_launch",
            Self::CpuActivity => "cpu_activity",
            Self::DeviceActivity => "device_activity",
            Self::Copy => "copy",
            Self::Sync => "sync",
            Self::Allocation => "allocation",
            Self::Eviction => "eviction",
            Self::Prefetch => "prefetch",
            Self::Stall => "stall",
            Self::Transport => "transport",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LedgerEvent {
    pub kind: LedgerEventKind,
    pub sync_class: Option<SyncClass>,
    pub metric_source: MetricSource,
    pub block_id: Option<ResidentBlockId>,
    pub from_tier: Option<MemoryTier>,
    pub to_tier: Option<MemoryTier>,
    pub bytes: usize,
    pub latency_ns: u64,
    pub label: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceTimelineSpan {
    pub device: DeviceOrdinal,
    pub start_ns: u64,
    pub end_ns: u64,
    pub metric_source: MetricSource,
    pub label: &'static str,
}

impl DeviceTimelineSpan {
    pub const fn new(
        device: DeviceOrdinal,
        start_ns: u64,
        end_ns: u64,
        metric_source: MetricSource,
        label: &'static str,
    ) -> Self {
        Self {
            device,
            start_ns,
            end_ns,
            metric_source,
            label,
        }
    }
}
