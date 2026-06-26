use crate::capabilities::json::json_opt_string;
use crate::engine::hot_path::status::HotPathGuardStatus;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HotPathGuardSummary {
    pub status: HotPathGuardStatus,
    pub token_index: u64,
    pub entered_scopes: u64,
    pub exited_scopes: u64,
    pub active_scopes_after_probe: u64,
    pub clean_scope_allocation_attempts: u64,
    pub deliberate_allocation_attempts: u64,
    pub deliberate_rejections: u64,
    pub ledger_allocation_events: u64,
    pub ledger_hot_path_allocations: u64,
    pub attempted_bytes: usize,
    pub release_to_system_calls: u64,
    pub usage_preserved_after_rejections: bool,
    pub error: Option<String>,
}

impl HotPathGuardSummary {
    pub fn passed(&self) -> bool {
        self.status == HotPathGuardStatus::Ok
            && self.entered_scopes == self.exited_scopes
            && self.active_scopes_after_probe == 0
            && self.clean_scope_allocation_attempts == 0
            && self.deliberate_allocation_attempts == self.deliberate_rejections
            && self.ledger_allocation_events == self.ledger_hot_path_allocations
            && self.ledger_hot_path_allocations == self.deliberate_rejections
            && self.release_to_system_calls == 0
            && self.usage_preserved_after_rejections
    }

    pub fn to_json(&self) -> String {
        format!(
            "{{\"status\":\"{}\",\"token_index\":{},\"entered_scopes\":{},\"exited_scopes\":{},\"active_scopes_after_probe\":{},\"clean_scope_allocation_attempts\":{},\"deliberate_allocation_attempts\":{},\"deliberate_rejections\":{},\"ledger_allocation_events\":{},\"ledger_hot_path_allocations\":{},\"attempted_bytes\":{},\"release_to_system_calls\":{},\"usage_preserved_after_rejections\":{},\"error\":{}}}",
            self.status.as_str(),
            self.token_index,
            self.entered_scopes,
            self.exited_scopes,
            self.active_scopes_after_probe,
            self.clean_scope_allocation_attempts,
            self.deliberate_allocation_attempts,
            self.deliberate_rejections,
            self.ledger_allocation_events,
            self.ledger_hot_path_allocations,
            self.attempted_bytes,
            self.release_to_system_calls,
            self.usage_preserved_after_rejections,
            json_opt_string(self.error.as_deref()),
        )
    }
}
