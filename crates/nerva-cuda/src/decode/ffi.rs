use std::os::raw::c_int;

pub(crate) const CUDA_ERROR_NO_DEVICE: i32 = 100;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaTinyDecodeResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) steps: u32,
    pub(crate) ring_capacity: u32,
    pub(crate) seed_token: u32,
    pub(crate) vocab_size: u32,
    pub(crate) hidden: u32,
    pub(crate) last_token: u32,
    pub(crate) graph_replays: u64,
    pub(crate) graph_nodes: u64,
    pub(crate) observed_tokens: u64,
    pub(crate) observed_token_hash: u64,
    pub(crate) token_ring_slots_touched: u64,
    pub(crate) token_ring_reuses: u64,
    pub(crate) token_ring_max_slot_version: u64,
    pub(crate) stale_tokens: u64,
    pub(crate) missing_tokens: u64,
    pub(crate) extra_tokens: u64,
    pub(crate) mismatched_tokens: u64,
    pub(crate) host_causality_edges: u64,
    pub(crate) resident_weight_bytes: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) h2d_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) graph_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) hot_path_allocations: u64,
    pub(crate) token_ledgers: u64,
    pub(crate) graph_replay_events: u64,
    pub(crate) device_activity_events: u64,
    pub(crate) copy_events: u64,
    pub(crate) soft_visibility_syncs: u64,
    pub(crate) hard_syncs: u64,
    pub(crate) host_event_wait_ns: u64,
    pub(crate) gpu_active_ns: u64,
    pub(crate) gpu_idle_ns: u64,
    pub(crate) wall_latency_ns: u64,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct NervaCudaHfDecodeStepRequest {
    pub(crate) dtype: u32,
    pub(crate) hidden: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) intermediate: u32,
    pub(crate) vocab_size: u32,
    pub(crate) position: u32,
    pub(crate) token_index: u64,
    pub(crate) rms_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) input: *const u16,
    pub(crate) rms_attn_weight: *const u16,
    pub(crate) rms_mlp_weight: *const u16,
    pub(crate) w_q: *const u16,
    pub(crate) w_k: *const u16,
    pub(crate) w_v: *const u16,
    pub(crate) w_o: *const u16,
    pub(crate) q_bias: *const u16,
    pub(crate) k_bias: *const u16,
    pub(crate) v_bias: *const u16,
    pub(crate) o_bias: *const u16,
    pub(crate) w_gate: *const u16,
    pub(crate) w_up: *const u16,
    pub(crate) w_down: *const u16,
    pub(crate) final_norm_weight: *const u16,
    pub(crate) lm_head: *const u16,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub(crate) struct NervaCudaHfDecodeStepResult {
    pub(crate) status: i32,
    pub(crate) cuda_error: i32,
    pub(crate) device_count: i32,
    pub(crate) dtype: u32,
    pub(crate) hidden: u32,
    pub(crate) heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) head_dim: u32,
    pub(crate) intermediate: u32,
    pub(crate) vocab_size: u32,
    pub(crate) token_index: u64,
    pub(crate) token: u32,
    pub(crate) slot_version: u64,
    pub(crate) completion: u32,
    pub(crate) output_hash: u64,
    pub(crate) resident_weight_bytes: u64,
    pub(crate) device_arena_bytes: u64,
    pub(crate) pinned_host_bytes: u64,
    pub(crate) h2d_bytes: u64,
    pub(crate) d2h_bytes: u64,
    pub(crate) kernel_launches: u64,
    pub(crate) sync_calls: u64,
    pub(crate) hot_path_allocations: u64,
}

unsafe extern "C" {
    fn nerva_cuda_hf_decode_step_u16(
        request: *const NervaCudaHfDecodeStepRequest,
        out: *mut NervaCudaHfDecodeStepResult,
    ) -> c_int;

    fn nerva_cuda_tiny_decode_smoke(
        steps: u32,
        ring_capacity: u32,
        seed_token: u32,
        out: *mut NervaCudaTinyDecodeResult,
    ) -> c_int;
}

pub(crate) fn run_hf_decode_step_u16(
    request: &NervaCudaHfDecodeStepRequest,
    out: &mut NervaCudaHfDecodeStepResult,
) -> c_int {
    unsafe { nerva_cuda_hf_decode_step_u16(request, out) }
}

pub(crate) fn run_tiny_decode_smoke(
    steps: u32,
    ring_capacity: u32,
    seed_token: u32,
    out: &mut NervaCudaTinyDecodeResult,
) -> c_int {
    unsafe { nerva_cuda_tiny_decode_smoke(steps, ring_capacity, seed_token, out) }
}
