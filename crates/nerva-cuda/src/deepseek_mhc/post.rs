use crate::deepseek_mhc::ffi::{
    NervaCudaDeepSeekMhcPostRequest, NervaCudaDeepSeekMhcPostResult, run_deepseek_mhc_post,
};
use crate::json::json_opt_str;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

const MAX_HC_MULT: u32 = 8;
const MAX_HC_HIDDEN: usize = 65_536;

#[derive(Clone, Debug)]
pub struct CudaDeepSeekMhcPostInput<'a> {
    pub tokens: u32,
    pub hc_mult: u32,
    pub hidden_size: u32,
    pub x: &'a [f32],
    pub residual: &'a [f32],
    pub post_layer_mix: &'a [f32],
    pub comb_res_mix: &'a [f32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekMhcPostSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub mhc_error: i32,
    pub tokens: u32,
    pub hc_mult: u32,
    pub hidden_size: u32,
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

impl CudaDeepSeekMhcPostSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"return_code\":{},\"cuda_error\":{},\"mhc_error\":{},\"tokens\":{},\"hc_mult\":{},\"hidden_size\":{},\"output\":{},\"output_hash\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.return_code,
            self.cuda_error,
            self.mhc_error,
            self.tokens,
            self.hc_mult,
            self.hidden_size,
            json_f32_array(&self.output),
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

pub fn deepseek_mhc_post(input: CudaDeepSeekMhcPostInput<'_>) -> CudaDeepSeekMhcPostSummary {
    let output_values = output_values(&input)
        .filter(|values| *values <= u32::MAX as usize)
        .unwrap_or(0);
    if !valid_shape(&input) {
        return failed_summary(
            &input,
            vec![0.0; output_values],
            "invalid DeepSeek mHC post shape",
        );
    }

    let mut output = vec![0.0f32; output_values];
    let request = NervaCudaDeepSeekMhcPostRequest {
        tokens: input.tokens,
        hc_mult: input.hc_mult,
        hidden_size: input.hidden_size,
        x: input.x.as_ptr(),
        residual: input.residual.as_ptr(),
        post_layer_mix: input.post_layer_mix.as_ptr(),
        comb_res_mix: input.comb_res_mix.as_ptr(),
        output: output.as_mut_ptr(),
    };
    let mut out = NervaCudaDeepSeekMhcPostResult::default();
    let return_code = run_deepseek_mhc_post(&request, &mut out);
    summarize(return_code, out, output)
}

fn valid_shape(input: &CudaDeepSeekMhcPostInput<'_>) -> bool {
    let Some(output_values) = output_values(input) else {
        return false;
    };
    let Some(hidden_values) = (input.tokens as usize).checked_mul(input.hidden_size as usize)
    else {
        return false;
    };
    let Some(hc_hidden_values) = (input.hc_mult as usize).checked_mul(input.hidden_size as usize)
    else {
        return false;
    };
    let Some(residual_values) = (input.tokens as usize).checked_mul(hc_hidden_values) else {
        return false;
    };
    let Some(post_values) = (input.tokens as usize).checked_mul(input.hc_mult as usize) else {
        return false;
    };
    let Some(comb_values) = post_values.checked_mul(input.hc_mult as usize) else {
        return false;
    };
    input.tokens > 0
        && input.hc_mult > 0
        && input.hc_mult <= MAX_HC_MULT
        && input.hidden_size > 0
        && hc_hidden_values <= MAX_HC_HIDDEN
        && hidden_values <= u32::MAX as usize
        && residual_values <= u32::MAX as usize
        && post_values <= u32::MAX as usize
        && comb_values <= u32::MAX as usize
        && output_values <= u32::MAX as usize
        && input.x.len() == hidden_values
        && input.residual.len() == residual_values
        && input.post_layer_mix.len() == post_values
        && input.comb_res_mix.len() == comb_values
}

fn output_values(input: &CudaDeepSeekMhcPostInput<'_>) -> Option<usize> {
    (input.tokens as usize)
        .checked_mul(input.hc_mult as usize)?
        .checked_mul(input.hidden_size as usize)
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekMhcPostResult,
    output: Vec<f32>,
) -> CudaDeepSeekMhcPostSummary {
    let status = if return_code == 0 && out.status == 0 && out.mhc_error == 0 {
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
            "CUDA DeepSeek mHC post failed: return_code={} status={} cuda_error={} mhc_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.mhc_error, out.device_count
        ))
    };
    CudaDeepSeekMhcPostSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        mhc_error: out.mhc_error,
        tokens: out.tokens,
        hc_mult: out.hc_mult,
        hidden_size: out.hidden_size,
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
    input: &CudaDeepSeekMhcPostInput<'_>,
    output: Vec<f32>,
    error: impl Into<String>,
) -> CudaDeepSeekMhcPostSummary {
    CudaDeepSeekMhcPostSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        mhc_error: -1,
        tokens: input.tokens,
        hc_mult: input.hc_mult,
        hidden_size: input.hidden_size,
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
