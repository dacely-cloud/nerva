#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SharedQueueProbeStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SharedQueueProbeSummary {
    pub status: SharedQueueProbeStatus,
    pub queue_capacity: u64,
    pub queue_blocks_ready: u64,
    pub atomic_control_blocks: u64,
    pub descriptors_posted: u64,
    pub descriptors_completed: u64,
    pub completion_records: u64,
    pub queue_full_rejections: u64,
    pub wrong_producer_rejections: u64,
    pub wrong_consumer_rejections: u64,
    pub bulk_payload_rejections: u64,
    pub payload_bytes_in_queue: u64,
    pub referenced_block_bytes: u64,
    pub phase_handoff_syncs: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl SharedQueueProbeSummary {
    pub fn passed(self) -> bool {
        matches!(self.status, SharedQueueProbeStatus::Ok)
            && self.queue_capacity > 0
            && self.queue_blocks_ready == 2
            && self.atomic_control_blocks == 2
            && self.descriptors_posted == self.queue_capacity
            && self.descriptors_completed == self.descriptors_posted
            && self.completion_records == self.descriptors_completed
            && self.queue_full_rejections == 1
            && self.wrong_producer_rejections == 1
            && self.wrong_consumer_rejections == 1
            && self.bulk_payload_rejections == 1
            && self.payload_bytes_in_queue == 0
            && self.referenced_block_bytes > 0
            && self.phase_handoff_syncs == self.descriptors_completed
            && self.hot_path_allocations == 0
    }

    pub fn to_json(self) -> String {
        let status = match self.status {
            SharedQueueProbeStatus::Ok => "ok",
            SharedQueueProbeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"queue_capacity\":{},\"queue_blocks_ready\":{},\"atomic_control_blocks\":{},\"descriptors_posted\":{},\"descriptors_completed\":{},\"completion_records\":{},\"queue_full_rejections\":{},\"wrong_producer_rejections\":{},\"wrong_consumer_rejections\":{},\"bulk_payload_rejections\":{},\"payload_bytes_in_queue\":{},\"referenced_block_bytes\":{},\"phase_handoff_syncs\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.queue_capacity,
            self.queue_blocks_ready,
            self.atomic_control_blocks,
            self.descriptors_posted,
            self.descriptors_completed,
            self.completion_records,
            self.queue_full_rejections,
            self.wrong_producer_rejections,
            self.wrong_consumer_rejections,
            self.bulk_payload_rejections,
            self.payload_bytes_in_queue,
            self.referenced_block_bytes,
            self.phase_handoff_syncs,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}

fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{value}\""))
}
