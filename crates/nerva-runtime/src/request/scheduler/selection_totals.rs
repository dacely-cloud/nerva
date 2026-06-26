use crate::request::scheduler::selection::SchedulerSelection;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct SchedulerSelectionTotals {
    pub(crate) decisions: u64,
    pub(crate) scanned_slots: u64,
    pub(crate) skipped_slots: u64,
    pub(crate) wraps: u64,
    pub(crate) no_ready_rejections: u64,
}

impl SchedulerSelectionTotals {
    pub(crate) fn record(&mut self, selection: SchedulerSelection) {
        self.decisions += 1;
        self.scanned_slots += selection.scanned_slots as u64;
        self.skipped_slots += selection.skipped_slots as u64;
        self.wraps += selection.wrapped as u64;
    }

    pub(crate) fn record_no_ready(&mut self) {
        self.no_ready_rejections += 1;
    }
}
