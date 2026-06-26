use crate::json::json_opt_str;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaTinyBlockSummary {
    pub status: SmokeStatus,
    pub hidden: u32,
    pub intermediate: u32,
    pub output: [u16; 2],
    pub output_hash: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub d2h_bytes: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaLoadedTinyBlockSummary {
    pub status: SmokeStatus,
    pub hidden: u32,
    pub intermediate: u32,
    pub output: [u16; 2],
    pub output_hash: u64,
    pub resident_weight_bytes: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaTinyBlockSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"intermediate\":{},\"output_bits\":[{},{}],\"output_hash\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"D2H_bytes\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.hidden,
            self.intermediate,
            self.output[0],
            self.output[1],
            self.output_hash,
            self.device_arena_bytes,
            self.pinned_host_bytes,
            self.kernel_launches,
            self.sync_calls,
            self.d2h_bytes,
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
    }

    pub(crate) fn unavailable(error: impl Into<String>) -> Self {
        Self::empty(SmokeStatus::Unavailable, error)
    }

    pub(crate) fn failed(error: impl Into<String>) -> Self {
        Self::empty(SmokeStatus::Failed, error)
    }

    fn empty(status: SmokeStatus, error: impl Into<String>) -> Self {
        Self {
            status,
            hidden: 2,
            intermediate: 2,
            output: [0, 0],
            output_hash: 0,
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            kernel_launches: 0,
            sync_calls: 0,
            d2h_bytes: 0,
            hot_path_allocations: 0,
            error: Some(error.into()),
        }
    }
}

impl CudaLoadedTinyBlockSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"intermediate\":{},\"output_bits\":[{},{}],\"output_hash\":{},\"resident_weight_bytes\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.hidden,
            self.intermediate,
            self.output[0],
            self.output[1],
            self.output_hash,
            self.resident_weight_bytes,
            self.device_arena_bytes,
            self.pinned_host_bytes,
            self.h2d_bytes,
            self.d2h_bytes,
            self.kernel_launches,
            self.sync_calls,
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
    }

    pub(crate) fn unavailable(error: impl Into<String>) -> Self {
        Self::empty(SmokeStatus::Unavailable, error)
    }

    pub(crate) fn failed(error: impl Into<String>) -> Self {
        Self::empty(SmokeStatus::Failed, error)
    }

    fn empty(status: SmokeStatus, error: impl Into<String>) -> Self {
        Self {
            status,
            hidden: 2,
            intermediate: 2,
            output: [0, 0],
            output_hash: 0,
            resident_weight_bytes: 0,
            device_arena_bytes: 0,
            pinned_host_bytes: 0,
            h2d_bytes: 0,
            d2h_bytes: 0,
            kernel_launches: 0,
            sync_calls: 0,
            hot_path_allocations: 0,
            error: Some(error.into()),
        }
    }
}
