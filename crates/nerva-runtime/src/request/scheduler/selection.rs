use nerva_core::types::id::request::RequestId;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SchedulerSelectionOutcome {
    Ready(SchedulerSelection),
    NoReady(SchedulerSelectionMiss),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SchedulerSelection {
    pub slot: usize,
    pub request_id: RequestId,
    pub scanned_slots: usize,
    pub skipped_slots: usize,
    pub wrapped: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SchedulerSelectionMiss {
    pub scanned_slots: usize,
    pub skipped_slots: usize,
    pub wrapped: bool,
}
