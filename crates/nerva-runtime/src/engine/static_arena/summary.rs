#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticArenaProbeSummary {
    pub device_capacity_bytes: usize,
    pub pinned_host_capacity_bytes: usize,
    pub host_capacity_bytes: usize,
    pub device_used_bytes: usize,
    pub pinned_host_used_bytes: usize,
    pub host_used_bytes: usize,
    pub bootstrap_blocks: usize,
    pub ready_blocks: usize,
    pub hot_path_rejections: u64,
    pub hot_path_allocation_attempts: u64,
    pub usage_preserved_after_rejections: bool,
}

impl StaticArenaProbeSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"device_capacity_bytes\":{},\"pinned_host_capacity_bytes\":{},\"host_capacity_bytes\":{},\"device_used_bytes\":{},\"pinned_host_used_bytes\":{},\"host_used_bytes\":{},\"bootstrap_blocks\":{},\"ready_blocks\":{},\"hot_path_rejections\":{},\"hot_path_allocation_attempts\":{},\"usage_preserved_after_rejections\":{}}}",
            self.device_capacity_bytes,
            self.pinned_host_capacity_bytes,
            self.host_capacity_bytes,
            self.device_used_bytes,
            self.pinned_host_used_bytes,
            self.host_used_bytes,
            self.bootstrap_blocks,
            self.ready_blocks,
            self.hot_path_rejections,
            self.hot_path_allocation_attempts,
            self.usage_preserved_after_rejections,
        )
    }
}
