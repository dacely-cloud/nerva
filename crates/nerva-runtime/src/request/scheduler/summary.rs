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
    pub completed_requests: usize,
    pub full_rejections: u64,
    pub duplicate_rejections: u64,
    pub missing_request_rejections: u64,
    pub scheduler_iterations: u64,
    pub max_active_requests: usize,
    pub host_observed_tokens: u64,
    pub generated_tokens: u64,
    pub bounded_slots: bool,
    pub unbounded_queue_ops: u64,
    pub hot_path_allocations: u64,
}

impl RequestSchedulerSummary {
    pub fn passed(&self) -> bool {
        self.capacity == 2
            && self.admitted_requests == 2
            && self.completed_requests == 2
            && self.full_rejections == 1
            && self.duplicate_rejections == 1
            && self.missing_request_rejections == 1
            && self.generated_tokens == self.host_observed_tokens
            && self.bounded_slots
            && self.unbounded_queue_ops == 0
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            RequestSchedulerProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"capacity\":{},\"admitted_requests\":{},\"active_requests\":{},\"completed_requests\":{},\"full_rejections\":{},\"duplicate_rejections\":{},\"missing_request_rejections\":{},\"scheduler_iterations\":{},\"max_active_requests\":{},\"host_observed_tokens\":{},\"generated_tokens\":{},\"bounded_slots\":{},\"unbounded_queue_ops\":{},\"hot_path_allocations\":{}}}",
            status,
            self.capacity,
            self.admitted_requests,
            self.active_requests,
            self.completed_requests,
            self.full_rejections,
            self.duplicate_rejections,
            self.missing_request_rejections,
            self.scheduler_iterations,
            self.max_active_requests,
            self.host_observed_tokens,
            self.generated_tokens,
            self.bounded_slots,
            self.unbounded_queue_ops,
            self.hot_path_allocations,
        )
    }
}
