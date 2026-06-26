use nerva_core::types::error::Result;
use nerva_core::types::id::device::DeviceOrdinal;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::critical::TokenCriticalPathReport;
use nerva_ledger::types::token::ledger::TokenLedger;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SchedulerLedgerTotals {
    pub(crate) token_ledgers: u64,
    pub(crate) critical_path_reports: u64,
    pub(crate) graph_replay_events: u64,
    pub(crate) device_activity_events: u64,
    pub(crate) copy_events: u64,
    pub(crate) soft_visibility_syncs: u64,
    pub(crate) host_event_wait_ns: u64,
    pub(crate) gpu_idle_ns: u64,
    pub(crate) estimated_events: u64,
    pub(crate) runtime_timestamp_events: u64,
    pub(crate) unclassified_syncs: u64,
    pub(crate) hot_path_allocations: u64,
    pub(crate) host_wait_gpu_idle_separated: bool,
}

impl SchedulerLedgerTotals {
    pub(crate) fn record(
        &mut self,
        ledger: &TokenLedger,
        report: &TokenCriticalPathReport,
        device: DeviceOrdinal,
    ) -> Result<()> {
        self.token_ledgers += 1;
        self.critical_path_reports += 1;
        self.graph_replay_events += ledger.event_count(LedgerEventKind::GraphReplay);
        self.device_activity_events += ledger.event_count(LedgerEventKind::DeviceActivity);
        self.copy_events += ledger.event_count(LedgerEventKind::Copy);
        self.soft_visibility_syncs += ledger.sync_count_for(SyncClass::SoftVisibilitySync);
        self.host_event_wait_ns += report.host_event_wait_ns;
        self.gpu_idle_ns += ledger.device_idle_ns(device)?;
        self.estimated_events += ledger.event_count_for_source(MetricSource::EstimatedModel);
        self.runtime_timestamp_events +=
            ledger.event_count_for_source(MetricSource::RuntimeTimestamp);
        self.unclassified_syncs += ledger
            .events
            .iter()
            .filter(|event| event.kind == LedgerEventKind::Sync && event.sync_class.is_none())
            .count() as u64;
        self.hot_path_allocations += ledger.hot_path_allocations;
        self.host_wait_gpu_idle_separated |= report.host_wait_gpu_idle_sources_separate;
        Ok(())
    }
}
