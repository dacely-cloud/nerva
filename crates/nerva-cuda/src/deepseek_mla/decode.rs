use crate::deepseek_mla::ffi::{
    NervaCudaDeepSeekMlaDecodeRequest, NervaCudaDeepSeekMlaDecodeResult, run_deepseek_mla_decode,
};
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekMlaDecodeSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub decode_error: i32,
    pub heads: u32,
    pub tokens: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub softmax_scale: f32,
    pub output: Vec<f32>,
    pub output_hash: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CudaDeepSeekMlaDecodeInput<'a> {
    pub heads: u32,
    pub tokens: u32,
    pub kv_lora_rank: u32,
    pub qk_nope_head_dim: u32,
    pub qk_rope_head_dim: u32,
    pub v_head_dim: u32,
    pub softmax_scale: f32,
    pub q_nope: &'a [f32],
    pub q_pe: &'a [f32],
    pub kv_c: &'a [f32],
    pub k_pe: &'a [f32],
    pub w_uk: &'a [f32],
    pub w_uv: &'a [f32],
}

pub fn deepseek_mla_decode(input: CudaDeepSeekMlaDecodeInput<'_>) -> CudaDeepSeekMlaDecodeSummary {
    let output_values = input.heads as usize * input.v_head_dim as usize;
    if !valid_shape(&input) {
        return failed_summary(
            &input,
            vec![0.0; output_values],
            "invalid DeepSeek MLA decode shape",
        );
    }

    let mut output = vec![0.0f32; output_values];
    let mut out = NervaCudaDeepSeekMlaDecodeResult::default();
    let request = NervaCudaDeepSeekMlaDecodeRequest {
        heads: input.heads,
        tokens: input.tokens,
        kv_lora_rank: input.kv_lora_rank,
        qk_nope_head_dim: input.qk_nope_head_dim,
        qk_rope_head_dim: input.qk_rope_head_dim,
        v_head_dim: input.v_head_dim,
        softmax_scale: input.softmax_scale,
        q_nope: input.q_nope.as_ptr(),
        q_pe: input.q_pe.as_ptr(),
        kv_c: input.kv_c.as_ptr(),
        k_pe: input.k_pe.as_ptr(),
        w_uk: input.w_uk.as_ptr(),
        w_uv: input.w_uv.as_ptr(),
        output: output.as_mut_ptr(),
    };
    let return_code = run_deepseek_mla_decode(&request, &mut out);
    summarize(return_code, out, output)
}

fn valid_shape(input: &CudaDeepSeekMlaDecodeInput<'_>) -> bool {
    input.heads > 0
        && input.tokens > 0
        && input.kv_lora_rank > 0
        && input.qk_nope_head_dim > 0
        && input.v_head_dim > 0
        && input.q_nope.len() == input.heads as usize * input.qk_nope_head_dim as usize
        && input.q_pe.len() == input.heads as usize * input.qk_rope_head_dim as usize
        && input.kv_c.len() == input.tokens as usize * input.kv_lora_rank as usize
        && input.k_pe.len() == input.tokens as usize * input.qk_rope_head_dim as usize
        && input.w_uk.len()
            == input.kv_lora_rank as usize * input.heads as usize * input.qk_nope_head_dim as usize
        && input.w_uv.len()
            == input.kv_lora_rank as usize * input.heads as usize * input.v_head_dim as usize
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekMlaDecodeResult,
    output: Vec<f32>,
) -> CudaDeepSeekMlaDecodeSummary {
    let status = if return_code == 0 && out.status == 0 && out.decode_error == 0 {
        SmokeStatus::Ok
    } else if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        SmokeStatus::Unavailable
    } else {
        SmokeStatus::Failed
    };
    let error = if status == SmokeStatus::Ok {
        None
    } else {
        Some(format!(
            "CUDA DeepSeek MLA decode failed: return_code={} status={} cuda_error={} decode_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.decode_error, out.device_count
        ))
    };
    CudaDeepSeekMlaDecodeSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        decode_error: out.decode_error,
        heads: out.heads,
        tokens: out.tokens,
        kv_lora_rank: out.kv_lora_rank,
        qk_nope_head_dim: out.qk_nope_head_dim,
        qk_rope_head_dim: out.qk_rope_head_dim,
        v_head_dim: out.v_head_dim,
        softmax_scale: out.softmax_scale,
        output,
        output_hash: out.output_hash,
        device_arena_bytes: out.device_arena_bytes,
        pinned_host_bytes: out.pinned_host_bytes,
        h2d_bytes: out.h2d_bytes,
        d2h_bytes: out.d2h_bytes,
        kernel_launches: out.kernel_launches,
        sync_calls: out.sync_calls,
        hot_path_allocations: out.hot_path_allocations,
        error,
    }
}

fn failed_summary(
    input: &CudaDeepSeekMlaDecodeInput<'_>,
    output: Vec<f32>,
    error: impl Into<String>,
) -> CudaDeepSeekMlaDecodeSummary {
    CudaDeepSeekMlaDecodeSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        decode_error: -1,
        heads: input.heads,
        tokens: input.tokens,
        kv_lora_rank: input.kv_lora_rank,
        qk_nope_head_dim: input.qk_nope_head_dim,
        qk_rope_head_dim: input.qk_rope_head_dim,
        v_head_dim: input.v_head_dim,
        softmax_scale: input.softmax_scale,
        output,
        output_hash: 0,
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
