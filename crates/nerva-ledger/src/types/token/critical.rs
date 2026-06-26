use nerva_core::types::error::Result;
use nerva_core::types::id::DeviceOrdinal;

use crate::types::event::LedgerEventKind;
use crate::types::metric::MetricSource;
use crate::types::sync::SyncClass;
use crate::types::token::ledger::TokenLedger;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenCriticalPathReport {
    pub token_index: u64,
    pub wall_latency_ns: u64,
    pub graph_replay_ns: u64,
    pub kernel_launch_ns: u64,
    pub cpu_active_ns: u64,
    pub device_activity_event_ns: u64,
    pub copy_ns: u64,
    pub sync_ns: u64,
    pub hard_sync_ns: u64,
    pub host_event_wait_ns: u64,
    pub policy_sync_ns: u64,
    pub phase_handoff_ns: u64,
    pub debug_sync_ns: u64,
    pub device_timeline_active_ns: u64,
    pub gpu_idle_ns: u64,
    pub host_wait_events: u64,
    pub device_timeline_spans: u64,
    pub estimated_event_count: u64,
    pub runtime_timestamp_event_count: u64,
    pub gpu_event_count: u64,
    pub hardware_counter_event_count: u64,
    pub profiler_event_count: u64,
    pub transport_completion_event_count: u64,
    pub estimated_latency_ns: u64,
    pub measured_latency_ns: u64,
    pub host_wait_gpu_idle_sources_separate: bool,
    pub host_wait_equals_gpu_idle_value: bool,
    pub estimated_presented_as_measured: bool,
}

impl TokenCriticalPathReport {
    pub fn from_ledger(ledger: &TokenLedger, device: DeviceOrdinal) -> Result<Self> {
        let device_timeline_active_ns = ledger.device_active_ns(device)?;
        let gpu_idle_ns = ledger.device_idle_ns(device)?;
        let estimated_event_count = ledger.event_count_for_source(MetricSource::EstimatedModel);
        let runtime_timestamp_event_count =
            ledger.event_count_for_source(MetricSource::RuntimeTimestamp);
        let gpu_event_count = ledger.event_count_for_source(MetricSource::GpuEvent);
        let hardware_counter_event_count =
            ledger.event_count_for_source(MetricSource::HardwareCounter);
        let profiler_event_count = ledger.event_count_for_source(MetricSource::Profiler);
        let transport_completion_event_count =
            ledger.event_count_for_source(MetricSource::TransportCompletion);
        let estimated_latency_ns = ledger.latency_ns_for_source(MetricSource::EstimatedModel);
        let measured_latency_ns = ledger
            .events
            .iter()
            .filter(|event| event.metric_source != MetricSource::EstimatedModel)
            .map(|event| event.latency_ns)
            .sum();
        let host_event_wait_ns = ledger.sync_latency_ns_for(SyncClass::SoftVisibilitySync);
        let host_wait_events = ledger.sync_count_for(SyncClass::SoftVisibilitySync);
        let device_timeline_spans = ledger
            .device_timeline
            .iter()
            .filter(|span| span.device == device)
            .count() as u64;

        Ok(Self {
            token_index: ledger.token_index,
            wall_latency_ns: ledger.total_latency_ns(),
            graph_replay_ns: ledger.latency_ns_for(LedgerEventKind::GraphReplay),
            kernel_launch_ns: ledger.latency_ns_for(LedgerEventKind::KernelLaunch),
            cpu_active_ns: ledger.latency_ns_for(LedgerEventKind::CpuActivity),
            device_activity_event_ns: ledger.latency_ns_for(LedgerEventKind::DeviceActivity),
            copy_ns: ledger.latency_ns_for(LedgerEventKind::Copy),
            sync_ns: ledger.latency_ns_for(LedgerEventKind::Sync),
            hard_sync_ns: ledger.sync_latency_ns_for(SyncClass::HardSync),
            host_event_wait_ns,
            policy_sync_ns: ledger.sync_latency_ns_for(SyncClass::PolicySync),
            phase_handoff_ns: ledger.sync_latency_ns_for(SyncClass::PhaseHandoff),
            debug_sync_ns: ledger.sync_latency_ns_for(SyncClass::DebugSync),
            device_timeline_active_ns,
            gpu_idle_ns,
            host_wait_events,
            device_timeline_spans,
            estimated_event_count,
            runtime_timestamp_event_count,
            gpu_event_count,
            hardware_counter_event_count,
            profiler_event_count,
            transport_completion_event_count,
            estimated_latency_ns,
            measured_latency_ns,
            host_wait_gpu_idle_sources_separate: host_wait_events > 0 && device_timeline_spans > 0,
            host_wait_equals_gpu_idle_value: host_event_wait_ns == gpu_idle_ns,
            estimated_presented_as_measured: false,
        })
    }

    pub fn proves_host_wait_not_gpu_idle(&self) -> bool {
        self.host_wait_gpu_idle_sources_separate
            && self.host_event_wait_ns > 0
            && self.host_event_wait_ns != self.gpu_idle_ns
            && !self.estimated_presented_as_measured
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"token_index\":{},\"wall_latency_ns\":{},\"graph_replay_ns\":{},\"kernel_launch_ns\":{},\"cpu_active_ns\":{},\"device_activity_event_ns\":{},\"copy_ns\":{},\"sync_ns\":{},\"hard_sync_ns\":{},\"host_event_wait_ns\":{},\"policy_sync_ns\":{},\"phase_handoff_ns\":{},\"debug_sync_ns\":{},\"device_timeline_active_ns\":{},\"gpu_idle_ns\":{},\"host_wait_events\":{},\"device_timeline_spans\":{},\"estimated_event_count\":{},\"runtime_timestamp_event_count\":{},\"gpu_event_count\":{},\"hardware_counter_event_count\":{},\"profiler_event_count\":{},\"transport_completion_event_count\":{},\"estimated_latency_ns\":{},\"measured_latency_ns\":{},\"host_wait_gpu_idle_sources_separate\":{},\"host_wait_equals_gpu_idle_value\":{},\"estimated_presented_as_measured\":{},\"proves_host_wait_not_gpu_idle\":{}}}",
            self.token_index,
            self.wall_latency_ns,
            self.graph_replay_ns,
            self.kernel_launch_ns,
            self.cpu_active_ns,
            self.device_activity_event_ns,
            self.copy_ns,
            self.sync_ns,
            self.hard_sync_ns,
            self.host_event_wait_ns,
            self.policy_sync_ns,
            self.phase_handoff_ns,
            self.debug_sync_ns,
            self.device_timeline_active_ns,
            self.gpu_idle_ns,
            self.host_wait_events,
            self.device_timeline_spans,
            self.estimated_event_count,
            self.runtime_timestamp_event_count,
            self.gpu_event_count,
            self.hardware_counter_event_count,
            self.profiler_event_count,
            self.transport_completion_event_count,
            self.estimated_latency_ns,
            self.measured_latency_ns,
            self.host_wait_gpu_idle_sources_separate,
            self.host_wait_equals_gpu_idle_value,
            self.estimated_presented_as_measured,
            self.proves_host_wait_not_gpu_idle(),
        )
    }
}
