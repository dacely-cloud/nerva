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
}

unsafe extern "C" {
    fn nerva_cuda_tiny_decode_smoke(
        steps: u32,
        ring_capacity: u32,
        seed_token: u32,
        out: *mut NervaCudaTinyDecodeResult,
    ) -> c_int;
}

pub(crate) fn run_tiny_decode_smoke(
    steps: u32,
    ring_capacity: u32,
    seed_token: u32,
    out: &mut NervaCudaTinyDecodeResult,
) -> c_int {
    unsafe { nerva_cuda_tiny_decode_smoke(steps, ring_capacity, seed_token, out) }
}
