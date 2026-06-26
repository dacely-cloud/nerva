use crate::transport::json::json_opt_static_str;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MemoryLoopStatus {
    Ok,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryLoopSummary {
    pub status: MemoryLoopStatus,
    pub tasks: u64,
    pub queue_capacity: usize,
    pub max_inflight: usize,
    pub queue_overflows: u64,
    pub disk_read_tasks: u64,
    pub prefetch_tasks: u64,
    pub staging_tasks: u64,
    pub eviction_tasks: u64,
    pub transport_prepare_tasks: u64,
    pub total_bytes: usize,
    pub total_predicted_visible_ns: u64,
    pub overlapped_ns: u64,
    pub actual_visible_ns: u64,
    pub ready_blocks: u64,
    pub prefetch_events: u64,
    pub eviction_events: u64,
    pub copy_events: u64,
    pub transport_events: u64,
    pub phase_handoff_syncs: u64,
    pub residency_decisions: u64,
    pub pageable_copies: u64,
    pub per_token_registrations: u64,
    pub page_faults: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl MemoryLoopSummary {
    pub fn passed(&self) -> bool {
        matches!(self.status, MemoryLoopStatus::Ok)
            && self.tasks > 0
            && self.queue_capacity > 0
            && self.max_inflight > 0
            && self.queue_overflows == 0
            && self.disk_read_tasks > 0
            && self.prefetch_tasks > 0
            && self.staging_tasks > 0
            && self.eviction_tasks > 0
            && self.transport_prepare_tasks > 0
            && self.total_bytes > 0
            && self.ready_blocks > 0
            && self.prefetch_events > 0
            && self.eviction_events > 0
            && self.copy_events > 0
            && self.transport_events > 0
            && self.phase_handoff_syncs > 0
            && self.residency_decisions == self.tasks
            && self.pageable_copies == 0
            && self.per_token_registrations == 0
            && self.page_faults == 0
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            MemoryLoopStatus::Ok => "ok",
            MemoryLoopStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"tasks\":{},\"queue_capacity\":{},\"max_inflight\":{},\"queue_overflows\":{},\"disk_read_tasks\":{},\"prefetch_tasks\":{},\"staging_tasks\":{},\"eviction_tasks\":{},\"transport_prepare_tasks\":{},\"total_bytes\":{},\"total_predicted_visible_ns\":{},\"overlapped_ns\":{},\"actual_visible_ns\":{},\"ready_blocks\":{},\"prefetch_events\":{},\"eviction_events\":{},\"copy_events\":{},\"transport_events\":{},\"phase_handoff_syncs\":{},\"residency_decisions\":{},\"pageable_copies\":{},\"per_token_registrations\":{},\"page_faults\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.tasks,
            self.queue_capacity,
            self.max_inflight,
            self.queue_overflows,
            self.disk_read_tasks,
            self.prefetch_tasks,
            self.staging_tasks,
            self.eviction_tasks,
            self.transport_prepare_tasks,
            self.total_bytes,
            self.total_predicted_visible_ns,
            self.overlapped_ns,
            self.actual_visible_ns,
            self.ready_blocks,
            self.prefetch_events,
            self.eviction_events,
            self.copy_events,
            self.transport_events,
            self.phase_handoff_syncs,
            self.residency_decisions,
            self.pageable_copies,
            self.per_token_registrations,
            self.page_faults,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}
