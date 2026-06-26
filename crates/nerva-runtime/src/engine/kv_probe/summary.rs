use crate::transport::json::json_opt_static_str;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KvResidencyProbeStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvResidencyProbeSummary {
    pub status: KvResidencyProbeStatus,
    pub pages: u32,
    pub page_bytes: usize,
    pub current_step: u64,
    pub hot_page_limit: usize,
    pub decisions: u64,
    pub keep_hot: u64,
    pub keep_warm: u64,
    pub prefetches: u64,
    pub demotions: u64,
    pub evictions: u64,
    pub copy_events: u64,
    pub prefetch_events: u64,
    pub eviction_events: u64,
    pub stall_events: u64,
    pub copy_bytes: usize,
    pub changed_bytes: usize,
    pub visible_stall_ns: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub vram_used_bytes: usize,
    pub dram_used_bytes: usize,
    pub error: Option<&'static str>,
}

impl KvResidencyProbeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            KvResidencyProbeStatus::Ok => "ok",
            KvResidencyProbeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"pages\":{},\"page_bytes\":{},\"current_step\":{},\"hot_page_limit\":{},\"decisions\":{},\"keep_hot\":{},\"keep_warm\":{},\"prefetches\":{},\"demotions\":{},\"evictions\":{},\"copy_events\":{},\"prefetch_events\":{},\"eviction_events\":{},\"stall_events\":{},\"copy_bytes\":{},\"changed_bytes\":{},\"visible_stall_ns\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"vram_used_bytes\":{},\"dram_used_bytes\":{},\"error\":{}}}",
            status,
            self.pages,
            self.page_bytes,
            self.current_step,
            self.hot_page_limit,
            self.decisions,
            self.keep_hot,
            self.keep_warm,
            self.prefetches,
            self.demotions,
            self.evictions,
            self.copy_events,
            self.prefetch_events,
            self.eviction_events,
            self.stall_events,
            self.copy_bytes,
            self.changed_bytes,
            self.visible_stall_ns,
            self.total_latency_ns,
            self.hot_path_allocations,
            self.vram_used_bytes,
            self.dram_used_bytes,
            json_opt_static_str(self.error),
        )
    }
}
