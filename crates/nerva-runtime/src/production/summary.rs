#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ProductionInvariantStatus {
    Ok,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionInvariantSummary {
    pub status: ProductionInvariantStatus,
    pub accepted_ledgers: u64,
    pub classified_sync_ledgers: u64,
    pub measured_fallbacks: u64,
    pub debug_sync_rejections: u64,
    pub debug_fallback_rejections: u64,
    pub unmeasured_fallback_rejections: u64,
    pub unnamed_fallback_rejections: u64,
    pub hot_path_allocations: u64,
    pub error: Option<&'static str>,
}

impl ProductionInvariantSummary {
    pub fn passed(&self) -> bool {
        matches!(self.status, ProductionInvariantStatus::Ok)
            && self.accepted_ledgers == 1
            && self.classified_sync_ledgers == 1
            && self.measured_fallbacks == 2
            && self.debug_sync_rejections == 1
            && self.debug_fallback_rejections == 1
            && self.unmeasured_fallback_rejections == 1
            && self.unnamed_fallback_rejections == 1
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            ProductionInvariantStatus::Ok => "ok",
            ProductionInvariantStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"accepted_ledgers\":{},\"classified_sync_ledgers\":{},\"measured_fallbacks\":{},\"debug_sync_rejections\":{},\"debug_fallback_rejections\":{},\"unmeasured_fallback_rejections\":{},\"unnamed_fallback_rejections\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.accepted_ledgers,
            self.classified_sync_ledgers,
            self.measured_fallbacks,
            self.debug_sync_rejections,
            self.debug_fallback_rejections,
            self.unmeasured_fallback_rejections,
            self.unnamed_fallback_rejections,
            self.hot_path_allocations,
            json_opt_static_str(self.error),
        )
    }
}

fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{value}\""))
}
