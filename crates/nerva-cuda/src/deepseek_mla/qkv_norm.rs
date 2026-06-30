use crate::deepseek_mla::ffi::{
    NervaCudaDeepSeekQKvRmsNormRequest, NervaCudaDeepSeekQKvRmsNormResult, run_deepseek_qkv_rmsnorm,
};
use crate::json::json_opt_str;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekQKvRmsNormSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub num_tokens: u32,
    pub q_size: u32,
    pub kv_size: u32,
    pub eps: f32,
    pub q_out: Vec<f32>,
    pub kv_out: Vec<f32>,
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

impl CudaDeepSeekQKvRmsNormSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"return_code\":{},\"cuda_error\":{},\"num_tokens\":{},\"q_size\":{},\"kv_size\":{},\"eps\":{},\"q_out\":{},\"kv_out\":{},\"output_hash\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.return_code,
            self.cuda_error,
            self.num_tokens,
            self.q_size,
            self.kv_size,
            self.eps,
            json_f32_array(&self.q_out),
            json_f32_array(&self.kv_out),
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

pub fn deepseek_qkv_rmsnorm(
    q: &[f32],
    kv: &[f32],
    q_weight: &[f32],
    kv_weight: &[f32],
    num_tokens: u32,
    q_size: u32,
    kv_size: u32,
    eps: f32,
) -> CudaDeepSeekQKvRmsNormSummary {
    let q_values = (num_tokens as usize)
        .checked_mul(q_size as usize)
        .unwrap_or(usize::MAX);
    let kv_values = (num_tokens as usize)
        .checked_mul(kv_size as usize)
        .unwrap_or(usize::MAX);
    if num_tokens == 0
        || q_size == 0
        || kv_size == 0
        || !eps.is_finite()
        || eps < 0.0
        || q_values == usize::MAX
        || kv_values == usize::MAX
        || q.len() != q_values
        || kv.len() != kv_values
        || q_weight.len() != q_size as usize
        || kv_weight.len() != kv_size as usize
        || q_values > u32::MAX as usize
        || kv_values > u32::MAX as usize
    {
        return failed_summary(
            num_tokens,
            q_size,
            kv_size,
            eps,
            Vec::new(),
            Vec::new(),
            "invalid DeepSeek Q/KV RMSNorm shape",
        );
    }

    let mut q_out = vec![0.0f32; q_values];
    let mut kv_out = vec![0.0f32; kv_values];
    let request = NervaCudaDeepSeekQKvRmsNormRequest {
        num_tokens,
        q_size,
        kv_size,
        eps,
        q: q.as_ptr(),
        kv: kv.as_ptr(),
        q_weight: q_weight.as_ptr(),
        kv_weight: kv_weight.as_ptr(),
        q_out: q_out.as_mut_ptr(),
        kv_out: kv_out.as_mut_ptr(),
    };
    let mut out = NervaCudaDeepSeekQKvRmsNormResult::default();
    let return_code = run_deepseek_qkv_rmsnorm(&request, &mut out);
    summarize(return_code, out, q_out, kv_out)
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekQKvRmsNormResult,
    q_out: Vec<f32>,
    kv_out: Vec<f32>,
) -> CudaDeepSeekQKvRmsNormSummary {
    let status = if return_code == 0 && out.status == 0 {
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
            "CUDA DeepSeek Q/KV RMSNorm failed: return_code={} status={} cuda_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.device_count
        ))
    };
    CudaDeepSeekQKvRmsNormSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        num_tokens: out.num_tokens,
        q_size: out.q_size,
        kv_size: out.kv_size,
        eps: out.eps,
        q_out,
        kv_out,
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
    num_tokens: u32,
    q_size: u32,
    kv_size: u32,
    eps: f32,
    q_out: Vec<f32>,
    kv_out: Vec<f32>,
    error: impl Into<String>,
) -> CudaDeepSeekQKvRmsNormSummary {
    CudaDeepSeekQKvRmsNormSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        num_tokens,
        q_size,
        kv_size,
        eps,
        q_out,
        kv_out,
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

fn json_f32_array(values: &[f32]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        if value.is_finite() {
            out.push_str(&value.to_string());
        } else {
            out.push_str("null");
        }
    }
    out.push(']');
    out
}
