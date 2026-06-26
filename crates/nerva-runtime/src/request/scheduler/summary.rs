#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RequestSchedulerProbeStatus {
    Ok,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestSchedulerSummary {
    pub status: RequestSchedulerProbeStatus,
    pub capacity: usize,
    pub admitted_requests: u64,
    pub active_requests: usize,
    pub completed_requests: u64,
    pub full_rejections: u64,
    pub duplicate_rejections: u64,
    pub missing_request_rejections: u64,
    pub premature_release_rejections: u64,
    pub released_slots: u64,
    pub reused_slots: u64,
    pub scheduler_iterations: u64,
    pub selection_decisions: u64,
    pub selection_scanned_slots: u64,
    pub selection_skipped_slots: u64,
    pub selection_wraps: u64,
    pub no_ready_selection_rejections: u64,
    pub max_active_requests: usize,
    pub host_observed_tokens: u64,
    pub generated_tokens: u64,
    pub token_ledgers: u64,
    pub critical_path_reports: u64,
    pub graph_replay_events: u64,
    pub device_activity_events: u64,
    pub copy_events: u64,
    pub soft_visibility_syncs: u64,
    pub host_event_wait_ns: u64,
    pub gpu_idle_ns: u64,
    pub estimated_events: u64,
    pub runtime_timestamp_events: u64,
    pub unclassified_syncs: u64,
    pub bounded_slots: bool,
    pub unbounded_queue_ops: u64,
    pub host_wait_gpu_idle_separated: bool,
    pub hot_path_allocations: u64,
}

impl RequestSchedulerSummary {
    pub fn passed(&self) -> bool {
        self.capacity == 2
            && self.admitted_requests == 3
            && self.completed_requests == 3
            && self.full_rejections == 1
            && self.duplicate_rejections == 1
            && self.missing_request_rejections == 1
            && self.premature_release_rejections == 1
            && self.released_slots == 3
            && self.reused_slots == 1
            && self.selection_decisions == self.generated_tokens
            && self.selection_scanned_slots >= self.selection_decisions
            && self.selection_skipped_slots > 0
            && self.no_ready_selection_rejections == 1
            && self.generated_tokens == self.host_observed_tokens
            && self.token_ledgers == self.generated_tokens
            && self.critical_path_reports == self.generated_tokens
            && self.graph_replay_events == self.generated_tokens
            && self.device_activity_events == self.generated_tokens
            && self.copy_events == self.generated_tokens
            && self.soft_visibility_syncs == self.generated_tokens
            && self.host_event_wait_ns > 0
            && self.gpu_idle_ns == 0
            && self.estimated_events > 0
            && self.runtime_timestamp_events > 0
            && self.unclassified_syncs == 0
            && self.bounded_slots
            && self.unbounded_queue_ops == 0
            && self.host_wait_gpu_idle_separated
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            RequestSchedulerProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"capacity\":{},\"admitted_requests\":{},\"active_requests\":{},\"completed_requests\":{},\"full_rejections\":{},\"duplicate_rejections\":{},\"missing_request_rejections\":{},\"premature_release_rejections\":{},\"released_slots\":{},\"reused_slots\":{},\"scheduler_iterations\":{},\"selection_decisions\":{},\"selection_scanned_slots\":{},\"selection_skipped_slots\":{},\"selection_wraps\":{},\"no_ready_selection_rejections\":{},\"max_active_requests\":{},\"host_observed_tokens\":{},\"generated_tokens\":{},\"token_ledgers\":{},\"critical_path_reports\":{},\"graph_replay_events\":{},\"device_activity_events\":{},\"copy_events\":{},\"soft_visibility_syncs\":{},\"host_event_wait_ns\":{},\"gpu_idle_ns\":{},\"estimated_events\":{},\"runtime_timestamp_events\":{},\"unclassified_syncs\":{},\"bounded_slots\":{},\"unbounded_queue_ops\":{},\"host_wait_gpu_idle_separated\":{},\"hot_path_allocations\":{}}}",
            status,
            self.capacity,
            self.admitted_requests,
            self.active_requests,
            self.completed_requests,
            self.full_rejections,
            self.duplicate_rejections,
            self.missing_request_rejections,
            self.premature_release_rejections,
            self.released_slots,
            self.reused_slots,
            self.scheduler_iterations,
            self.selection_decisions,
            self.selection_scanned_slots,
            self.selection_skipped_slots,
            self.selection_wraps,
            self.no_ready_selection_rejections,
            self.max_active_requests,
            self.host_observed_tokens,
            self.generated_tokens,
            self.token_ledgers,
            self.critical_path_reports,
            self.graph_replay_events,
            self.device_activity_events,
            self.copy_events,
            self.soft_visibility_syncs,
            self.host_event_wait_ns,
            self.gpu_idle_ns,
            self.estimated_events,
            self.runtime_timestamp_events,
            self.unclassified_syncs,
            self.bounded_slots,
            self.unbounded_queue_ops,
            self.host_wait_gpu_idle_separated,
            self.hot_path_allocations,
        )
    }
}
