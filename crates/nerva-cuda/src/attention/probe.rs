use crate::attention::ffi::{
    CUDA_ERROR_NO_DEVICE, NervaCudaTieredAttentionResult, run_tiered_attention_smoke,
};
use crate::attention::summary::CudaTieredAttentionSummary;
use crate::smoke::status::SmokeStatus;

pub fn tiered_attention_smoke() -> CudaTieredAttentionSummary {
    let mut out = NervaCudaTieredAttentionResult::default();
    let return_code = run_tiered_attention_smoke(&mut out);

    if return_code == 0
        && out.status == 0
        && out.hidden == 2
        && out.heads == 1
        && out.blocks == 2
        && out.tokens == 4
        && out.output_hash != 0
        && out.cpu_block_events == 1
        && out.device_block_events == 1
        && out.resident_kv_bytes > 0
        && out.h2d_bytes >= out.resident_kv_bytes
        && out.d2h_bytes > 0
        && out.kernel_launches == 1
        && out.hot_path_allocations == 0
        && out.output.iter().all(|value| value.is_finite())
    {
        return CudaTieredAttentionSummary {
            status: SmokeStatus::Ok,
            hidden: out.hidden,
            heads: out.heads,
            blocks: out.blocks,
            tokens: out.tokens,
            output: out.output,
            output_hash: out.output_hash,
            cpu_block_events: out.cpu_block_events,
            device_block_events: out.device_block_events,
            resident_kv_bytes: out.resident_kv_bytes,
            device_arena_bytes: out.device_arena_bytes,
            pinned_host_bytes: out.pinned_host_bytes,
            h2d_bytes: out.h2d_bytes,
            d2h_bytes: out.d2h_bytes,
            kernel_launches: out.kernel_launches,
            sync_calls: out.sync_calls,
            hot_path_allocations: out.hot_path_allocations,
            error: None,
        };
    }

    let reason = format!(
        "CUDA tiered attention smoke failed: return_code={} status={} cuda_error={} device_count={} hidden={} heads={} blocks={} tokens={} output_hash={} cpu_events={} device_events={} kernel_launches={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.hidden,
        out.heads,
        out.blocks,
        out.tokens,
        out.output_hash,
        out.cpu_block_events,
        out.device_block_events,
        out.kernel_launches,
    );
    if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        CudaTieredAttentionSummary::unavailable(reason)
    } else {
        CudaTieredAttentionSummary::failed(reason)
    }
}
