use crate::json::json_opt_str;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekKvSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub block_size: u32,
    pub token_index: u32,
    pub token_stride: u32,
    pub scale_dim: u32,
    pub block_bytes: u64,
    pub output_hash: u64,
    pub output: Vec<u8>,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaDeepSeekKvSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"return_code\":{},\"cuda_error\":{},\"block_size\":{},\"token_index\":{},\"token_stride\":{},\"scale_dim\":{},\"block_bytes\":{},\"output_hash\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.return_code,
            self.cuda_error,
            self.block_size,
            self.token_index,
            self.token_stride,
            self.scale_dim,
            self.block_bytes,
            self.output_hash,
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
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekCompressedSlotMappingSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub num_tokens: u32,
    pub num_reqs: u32,
    pub block_table_stride: u32,
    pub block_size: u32,
    pub compress_ratio: u32,
    pub valid_slots: u32,
    pub pad_slots: u32,
    pub output_hash: u64,
    pub output_slots: Vec<i64>,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaDeepSeekCompressedSlotMappingSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"return_code\":{},\"cuda_error\":{},\"num_tokens\":{},\"num_reqs\":{},\"block_table_stride\":{},\"block_size\":{},\"compress_ratio\":{},\"valid_slots\":{},\"pad_slots\":{},\"output_hash\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.return_code,
            self.cuda_error,
            self.num_tokens,
            self.num_reqs,
            self.block_table_stride,
            self.block_size,
            self.compress_ratio,
            self.valid_slots,
            self.pad_slots,
            self.output_hash,
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
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekC128TopkMetadataSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub num_tokens: u32,
    pub num_decode_tokens: u32,
    pub num_prefill_tokens: u32,
    pub num_reqs: u32,
    pub block_table_stride: u32,
    pub block_size: u32,
    pub compress_ratio: u32,
    pub max_compressed_tokens: u32,
    pub valid_decode_tokens: u32,
    pub decode_entries: u32,
    pub prefill_entries: u32,
    pub output_hash: u64,
    pub global_decode: Vec<i32>,
    pub decode_lens: Vec<i32>,
    pub prefill_local: Vec<i32>,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaDeepSeekC128TopkMetadataSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"return_code\":{},\"cuda_error\":{},\"num_tokens\":{},\"num_decode_tokens\":{},\"num_prefill_tokens\":{},\"num_reqs\":{},\"block_table_stride\":{},\"block_size\":{},\"compress_ratio\":{},\"max_compressed_tokens\":{},\"valid_decode_tokens\":{},\"decode_entries\":{},\"prefill_entries\":{},\"output_hash\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.return_code,
            self.cuda_error,
            self.num_tokens,
            self.num_decode_tokens,
            self.num_prefill_tokens,
            self.num_reqs,
            self.block_table_stride,
            self.block_size,
            self.compress_ratio,
            self.max_compressed_tokens,
            self.valid_decode_tokens,
            self.decode_entries,
            self.prefill_entries,
            self.output_hash,
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
}
