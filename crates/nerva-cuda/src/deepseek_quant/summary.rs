use crate::json::json_opt_str;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekQuantSummary {
    pub status: SmokeStatus,
    pub fp8_rows: u32,
    pub fp8_cols: u32,
    pub fp8_block_rows: u32,
    pub fp8_block_cols: u32,
    pub mxfp4_rows: u32,
    pub mxfp4_packed_cols: u32,
    pub mxfp4_scale_packed_cols: u32,
    pub fp8_output_hash: u64,
    pub mxfp4_output_hash: u64,
    pub fp8_mismatches: u64,
    pub mxfp4_mismatches: u64,
    pub fp8_max_abs_diff: f32,
    pub mxfp4_max_abs_diff: f32,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaDeepSeekQuantSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"fp8_rows\":{},\"fp8_cols\":{},\"fp8_block_rows\":{},\"fp8_block_cols\":{},\"mxfp4_rows\":{},\"mxfp4_packed_cols\":{},\"mxfp4_scale_packed_cols\":{},\"fp8_output_hash\":{},\"mxfp4_output_hash\":{},\"fp8_mismatches\":{},\"mxfp4_mismatches\":{},\"fp8_max_abs_diff\":{},\"mxfp4_max_abs_diff\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.fp8_rows,
            self.fp8_cols,
            self.fp8_block_rows,
            self.fp8_block_cols,
            self.mxfp4_rows,
            self.mxfp4_packed_cols,
            self.mxfp4_scale_packed_cols,
            self.fp8_output_hash,
            self.mxfp4_output_hash,
            self.fp8_mismatches,
            self.mxfp4_mismatches,
            self.fp8_max_abs_diff,
            self.mxfp4_max_abs_diff,
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
            fp8_rows: 3,
            fp8_cols: 4,
            fp8_block_rows: 2,
            fp8_block_cols: 2,
            mxfp4_rows: 2,
            mxfp4_packed_cols: 4,
            mxfp4_scale_packed_cols: 2,
            fp8_output_hash: 0,
            mxfp4_output_hash: 0,
            fp8_mismatches: 0,
            mxfp4_mismatches: 0,
            fp8_max_abs_diff: 0.0,
            mxfp4_max_abs_diff: 0.0,
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
