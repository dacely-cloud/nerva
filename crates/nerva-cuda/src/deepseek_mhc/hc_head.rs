use crate::deepseek_mhc::ffi::{
    NervaCudaDeepSeekMhcHeadRequest, NervaCudaDeepSeekMhcHeadResult, run_deepseek_mhc_head,
};
use crate::json::json_opt_str;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

const MAX_HC_MULT: u32 = 64;
const MAX_HC_HIDDEN: usize = 65_536;

#[derive(Clone, Debug)]
pub struct CudaDeepSeekMhcHeadInput<'a> {
    pub tokens: u32,
    pub hc_mult: u32,
    pub hidden_size: u32,
    pub rms_eps: f32,
    pub hc_eps: f32,
    pub hc_scale: f32,
    pub hidden_states: &'a [f32],
    pub fn_weights: &'a [f32],
    pub hc_base: &'a [f32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekMhcHeadSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub mhc_error: i32,
    pub tokens: u32,
    pub hc_mult: u32,
    pub hidden_size: u32,
    pub rms_eps: f32,
    pub hc_eps: f32,
    pub hc_scale: f32,
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

impl CudaDeepSeekMhcHeadSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"return_code\":{},\"cuda_error\":{},\"mhc_error\":{},\"tokens\":{},\"hc_mult\":{},\"hidden_size\":{},\"rms_eps\":{},\"hc_eps\":{},\"hc_scale\":{},\"output\":{},\"output_hash\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.return_code,
            self.cuda_error,
            self.mhc_error,
            self.tokens,
            self.hc_mult,
            self.hidden_size,
            self.rms_eps,
            self.hc_eps,
            self.hc_scale,
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

pub fn deepseek_mhc_head(input: CudaDeepSeekMhcHeadInput<'_>) -> CudaDeepSeekMhcHeadSummary {
    let output_values = output_values(&input)
        .filter(|values| *values <= u32::MAX as usize)
        .unwrap_or(0);
    if !valid_shape(&input) {
        return failed_summary(
            &input,
            vec![0.0; output_values],
            "invalid DeepSeek mHC head shape",
        );
    }

    let mut output = vec![0.0f32; output_values];
    let request = NervaCudaDeepSeekMhcHeadRequest {
        tokens: input.tokens,
        hc_mult: input.hc_mult,
        hidden_size: input.hidden_size,
        rms_eps: input.rms_eps,
        hc_eps: input.hc_eps,
        hc_scale: input.hc_scale,
        hidden_states: input.hidden_states.as_ptr(),
        fn_weights: input.fn_weights.as_ptr(),
        hc_base: input.hc_base.as_ptr(),
        output: output.as_mut_ptr(),
    };
    let mut out = NervaCudaDeepSeekMhcHeadResult::default();
    let return_code = run_deepseek_mhc_head(&request, &mut out);
    summarize(return_code, out, output)
}

fn valid_shape(input: &CudaDeepSeekMhcHeadInput<'_>) -> bool {
    let Some(output_values) = output_values(input) else {
        return false;
    };
    let Some(hc_hidden_values) = (input.hc_mult as usize).checked_mul(input.hidden_size as usize)
    else {
        return false;
    };
    let Some(hidden_values) = (input.tokens as usize).checked_mul(hc_hidden_values) else {
        return false;
    };
    let Some(fn_values) = (input.hc_mult as usize).checked_mul(hc_hidden_values) else {
        return false;
    };
    input.tokens > 0
        && input.hc_mult > 0
        && input.hc_mult <= MAX_HC_MULT
        && input.hidden_size > 0
        && input.rms_eps.is_finite()
        && input.rms_eps >= 0.0
        && input.hc_eps.is_finite()
        && input.hc_scale.is_finite()
        && output_values <= u32::MAX as usize
        && hc_hidden_values <= MAX_HC_HIDDEN
        && hidden_values <= u32::MAX as usize
        && fn_values <= u32::MAX as usize
        && input.hidden_states.len() == hidden_values
        && input.fn_weights.len() == fn_values
        && input.hc_base.len() == input.hc_mult as usize
}

fn output_values(input: &CudaDeepSeekMhcHeadInput<'_>) -> Option<usize> {
    (input.tokens as usize).checked_mul(input.hidden_size as usize)
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekMhcHeadResult,
    output: Vec<f32>,
) -> CudaDeepSeekMhcHeadSummary {
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
            "CUDA DeepSeek mHC head failed: return_code={} status={} cuda_error={} mhc_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.mhc_error, out.device_count
        ))
    };
    CudaDeepSeekMhcHeadSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        mhc_error: out.mhc_error,
        tokens: out.tokens,
        hc_mult: out.hc_mult,
        hidden_size: out.hidden_size,
        rms_eps: out.rms_eps,
        hc_eps: out.hc_eps,
        hc_scale: out.hc_scale,
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
    input: &CudaDeepSeekMhcHeadInput<'_>,
    output: Vec<f32>,
    error: impl Into<String>,
) -> CudaDeepSeekMhcHeadSummary {
    CudaDeepSeekMhcHeadSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        mhc_error: -1,
        tokens: input.tokens,
        hc_mult: input.hc_mult,
        hidden_size: input.hidden_size,
        rms_eps: input.rms_eps,
        hc_eps: input.hc_eps,
        hc_scale: input.hc_scale,
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
