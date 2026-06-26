use crate::transport::json::json_opt_static_str;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ExecutionTransactionStatus {
    Ok,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionTransactionSummary {
    pub status: ExecutionTransactionStatus,
    pub operations: u64,
    pub graph_capturable_operations: u64,
    pub cpu_operations: u64,
    pub gpu_operations: u64,
    pub block_uses: u64,
    pub block_version_dependencies: u64,
    pub execution_decisions: u64,
    pub hard_syncs: u64,
    pub soft_visibility_syncs: u64,
    pub phase_handoff_syncs: u64,
    pub policy_syncs: u64,
    pub debug_syncs: u64,
    pub graph_replay_events: u64,
    pub kernel_launch_events: u64,
    pub device_activity_events: u64,
    pub cpu_activity_events: u64,
    pub device_active_ns: u64,
    pub gpu_idle_ns: u64,
    pub host_event_wait_ns: u64,
    pub total_predicted_visible_ns: u64,
    pub hot_path_allocations: u64,
    pub stale_dependencies: u64,
    pub unclassified_syncs: u64,
    pub error: Option<&'static str>,
}

impl ExecutionTransactionSummary {
    pub fn passed(&self) -> bool {
        matches!(self.status, ExecutionTransactionStatus::Ok)
            && self.operations > 0
            && self.block_uses > 0
            && self.block_version_dependencies == self.block_uses
            && self.execution_decisions == self.operations
            && self.graph_capturable_operations > 0
            && self.gpu_operations > 0
            && self.hard_syncs > 0
            && self.soft_visibility_syncs > 0
            && self.phase_handoff_syncs > 0
            && self.debug_syncs == 0
            && self.device_active_ns > 0
            && self.host_event_wait_ns > 0
            && self.hot_path_allocations == 0
            && self.stale_dependencies == 0
            && self.unclassified_syncs == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            ExecutionTransactionStatus::Ok => "ok",
            ExecutionTransactionStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"operations\":{},\"graph_capturable_operations\":{},\"cpu_operations\":{},\"gpu_operations\":{},\"block_uses\":{},\"block_version_dependencies\":{},\"execution_decisions\":{},\"hard_syncs\":{},\"soft_visibility_syncs\":{},\"phase_handoff_syncs\":{},\"policy_syncs\":{},\"debug_syncs\":{},\"graph_replay_events\":{},\"kernel_launch_events\":{},\"device_activity_events\":{},\"cpu_activity_events\":{},\"device_active_ns\":{},\"gpu_idle_ns\":{},\"host_event_wait_ns\":{},\"total_predicted_visible_ns\":{},\"hot_path_allocations\":{},\"stale_dependencies\":{},\"unclassified_syncs\":{},\"error\":{}}}",
            status,
            self.operations,
            self.graph_capturable_operations,
            self.cpu_operations,
            self.gpu_operations,
            self.block_uses,
            self.block_version_dependencies,
            self.execution_decisions,
            self.hard_syncs,
            self.soft_visibility_syncs,
            self.phase_handoff_syncs,
            self.policy_syncs,
            self.debug_syncs,
            self.graph_replay_events,
            self.kernel_launch_events,
            self.device_activity_events,
            self.cpu_activity_events,
            self.device_active_ns,
            self.gpu_idle_ns,
            self.host_event_wait_ns,
            self.total_predicted_visible_ns,
            self.hot_path_allocations,
            self.stale_dependencies,
            self.unclassified_syncs,
            json_opt_static_str(self.error),
        )
    }
}
