#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TokenPolicyStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TokenPolicySummary {
    pub status: TokenPolicyStatus,
    pub steps: u64,
    pub device_fast_steps: u64,
    pub host_policy_steps: u64,
    pub hybrid_validation_steps: u64,
    pub seed_edges: u64,
    pub device_ring_edges: u64,
    pub host_causality_edges: u64,
    pub policy_syncs: u64,
    pub soft_visibility_syncs: u64,
    pub host_visibility_hard_dependencies: u64,
    pub device_fast_host_dependencies: u64,
    pub graph_replays: u64,
    pub observed_tokens: u64,
    pub mismatched_tokens: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl TokenPolicySummary {
    pub fn passed(self) -> bool {
        matches!(self.status, TokenPolicyStatus::Ok)
            && self.steps > 0
            && self.device_fast_steps > 0
            && self.host_policy_steps > 0
            && self.hybrid_validation_steps > 0
            && self.seed_edges == 1
            && self.device_ring_edges > 0
            && self.host_causality_edges == self.host_policy_steps
            && self.policy_syncs == self.host_policy_steps
            && self.soft_visibility_syncs == self.steps
            && self.host_visibility_hard_dependencies == self.host_policy_steps
            && self.device_fast_host_dependencies == 0
            && self.graph_replays == self.steps
            && self.observed_tokens == self.steps
            && self.mismatched_tokens == 0
            && self.hot_path_allocations == 0
    }

    pub fn to_json(self) -> String {
        let status = match self.status {
            TokenPolicyStatus::Ok => "ok",
            TokenPolicyStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"steps\":{},\"device_fast_steps\":{},\"host_policy_steps\":{},\"hybrid_validation_steps\":{},\"seed_edges\":{},\"device_ring_edges\":{},\"host_causality_edges\":{},\"policy_syncs\":{},\"soft_visibility_syncs\":{},\"host_visibility_hard_dependencies\":{},\"device_fast_host_dependencies\":{},\"graph_replays\":{},\"observed_tokens\":{},\"mismatched_tokens\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.steps,
            self.device_fast_steps,
            self.host_policy_steps,
            self.hybrid_validation_steps,
            self.seed_edges,
            self.device_ring_edges,
            self.host_causality_edges,
            self.policy_syncs,
            self.soft_visibility_syncs,
            self.host_visibility_hard_dependencies,
            self.device_fast_host_dependencies,
            self.graph_replays,
            self.observed_tokens,
            self.mismatched_tokens,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}

fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{value}\""))
}
