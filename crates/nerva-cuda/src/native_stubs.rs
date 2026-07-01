use std::ffi::c_void;
use std::os::raw::c_int;
use std::ptr;

const CUDA_ERROR_NO_DEVICE: i32 = 100;

fn unavailable(out: *mut c_void) -> c_int {
    if !out.is_null() {
        unsafe {
            let fields = out.cast::<i32>();
            fields.write(-1);
            fields.add(1).write(CUDA_ERROR_NO_DEVICE);
        }
    }
    -1
}

fn unavailable_status_only(out: *mut c_void) -> c_int {
    if !out.is_null() {
        unsafe {
            out.cast::<i32>().write(-1);
        }
    }
    -1
}

fn clear_session(session: *mut *mut c_void) {
    if !session.is_null() {
        unsafe {
            session.write(ptr::null_mut());
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_device_smoke(out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_synthetic_graph_smoke(
    _steps: u32,
    _ring_capacity: u32,
    _seed_token: u32,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_tiny_block_smoke(out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_loaded_tiny_block_smoke(out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_block_forward_u16(_request: *const c_void, out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_greedy_sampler_smoke(out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_sample_u16(_request: *const c_void, out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_step_u16(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_chain_u16(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_u16(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_plan_layout(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable_status_only(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_session_create(
    _request: *const c_void,
    out: *mut c_void,
    session: *mut *mut c_void,
) -> c_int {
    clear_session(session);
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_session_run(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_session_start(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_session_advance(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_deepseek_v4_swa_kv_snapshot(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_deepseek_v3_mla_kv_snapshot(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_deepseek_v32_mla_packed_kv_snapshot(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_deepseek_v32_indexer_kv_snapshot(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_deepseek_v32_indexer_query_state_snapshot(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_deepseek_v4_compressed_kv_snapshot(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_deepseek_v4_indexer_kv_snapshot(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_deepseek_v4_mhc_snapshot(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_projection_batch_plan(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_projection_batch_execute(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_layer_projection_batch_execute(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_batch_advance_one(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_session_fork_shared_weights(
    _request: *const c_void,
    out: *mut c_void,
    session: *mut *mut c_void,
) -> c_int {
    clear_session(session);
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_hf_decode_sequence_session_destroy(
    _session: *mut c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_tiny_decode_smoke(
    _steps: u32,
    _ring_capacity: u32,
    _seed_token: u32,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_tiered_attention_smoke(out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_backend_contract_smoke(
    out: *mut c_void,
    _device_bytes: u64,
    _pinned_bytes: u64,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_projection_bench(_request: *const c_void, out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_experimental_rt_candidate_bench(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_experimental_rt_cold_kv_staging_bench(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_quant_smoke(out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_quant_fp8_dequant(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_quant_mxfp4_dequant(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_quant_fp8_f32_scale_matvec(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_quant_fp8_f32_scale_encoded_matvec(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_quant_fp8_f32_scale_encoded_gemm_tokens(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_quant_fp8_e8m0_scale_encoded_gemm_tokens(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_fused_inv_rope_fp8_quant(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_router_smoke(out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_router_route(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_mla_smoke(out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_mla_decode(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_qkv_rmsnorm(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_mhc_pre(_request: *const c_void, out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_mhc_post(_request: *const c_void, out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_mhc_fused_post_pre(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_mhc_head(_request: *const c_void, out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_kv_fp8_ds_mla_pack(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_v32_kv_fp8_ds_mla_pack(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_compressed_slot_mapping(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_c128_topk_metadata(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_c4_indexer_topk(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_save_partial_states(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_compress_norm_rope_fp8_cache(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_moe_smoke(out: *mut c_void) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_moe_forward(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_megamoe_prepare(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn nerva_cuda_deepseek_megamoe_experts(
    _request: *const c_void,
    out: *mut c_void,
) -> c_int {
    unavailable(out)
}
