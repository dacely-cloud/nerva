use crate::deepseek_mhc::ffi::{
    NervaCudaDeepSeekMhcPreRequest, NervaCudaDeepSeekMhcPreResult, run_deepseek_mhc_pre,
};
use crate::json::json_opt_str;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

const MAX_HC_MULT: u32 = 8;
const MAX_HC_HIDDEN: usize = 65_536;

#[derive(Clone, Debug)]
pub struct CudaDeepSeekMhcPreInput<'a> {
    pub tokens: u32,
    pub hc_mult: u32,
    pub hidden_size: u32,
    pub sinkhorn_repeat: u32,
    pub rms_eps: f32,
    pub hc_pre_eps: f32,
    pub hc_sinkhorn_eps: f32,
    pub hc_post_mult_value: f32,
    pub residual: &'a [f32],
    pub fn_weights: &'a [f32],
    pub hc_scale: &'a [f32],
    pub hc_base: &'a [f32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekMhcPreSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub mhc_error: i32,
    pub tokens: u32,
    pub hc_mult: u32,
    pub hidden_size: u32,
    pub sinkhorn_repeat: u32,
    pub rms_eps: f32,
    pub hc_pre_eps: f32,
    pub hc_sinkhorn_eps: f32,
    pub hc_post_mult_value: f32,
    pub post_mix: Vec<f32>,
    pub comb_mix: Vec<f32>,
    pub layer_input: Vec<f32>,
    pub post_mix_hash: u64,
    pub comb_mix_hash: u64,
    pub layer_input_hash: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaDeepSeekMhcPreSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"return_code\":{},\"cuda_error\":{},\"mhc_error\":{},\"tokens\":{},\"hc_mult\":{},\"hidden_size\":{},\"sinkhorn_repeat\":{},\"rms_eps\":{},\"hc_pre_eps\":{},\"hc_sinkhorn_eps\":{},\"hc_post_mult_value\":{},\"post_mix\":{},\"comb_mix\":{},\"layer_input\":{},\"post_mix_hash\":{},\"comb_mix_hash\":{},\"layer_input_hash\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.return_code,
            self.cuda_error,
            self.mhc_error,
            self.tokens,
            self.hc_mult,
            self.hidden_size,
            self.sinkhorn_repeat,
            self.rms_eps,
            self.hc_pre_eps,
            self.hc_sinkhorn_eps,
            self.hc_post_mult_value,
            json_f32_array(&self.post_mix),
            json_f32_array(&self.comb_mix),
            json_f32_array(&self.layer_input),
            self.post_mix_hash,
            self.comb_mix_hash,
            self.layer_input_hash,
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

pub fn deepseek_mhc_pre(input: CudaDeepSeekMhcPreInput<'_>) -> CudaDeepSeekMhcPreSummary {
    let output_values = output_values(&input)
        .filter(|(post, comb, layer)| {
            *post <= u32::MAX as usize && *comb <= u32::MAX as usize && *layer <= u32::MAX as usize
        })
        .unwrap_or((0, 0, 0));
    if !valid_shape(&input) {
        return failed_summary(
            &input,
            vec![0.0; output_values.0],
            vec![0.0; output_values.1],
            vec![0.0; output_values.2],
            "invalid DeepSeek mHC pre shape",
        );
    }

    let mut post_mix = vec![0.0f32; output_values.0];
    let mut comb_mix = vec![0.0f32; output_values.1];
    let mut layer_input = vec![0.0f32; output_values.2];
    let request = NervaCudaDeepSeekMhcPreRequest {
        tokens: input.tokens,
        hc_mult: input.hc_mult,
        hidden_size: input.hidden_size,
        sinkhorn_repeat: input.sinkhorn_repeat,
        rms_eps: input.rms_eps,
        hc_pre_eps: input.hc_pre_eps,
        hc_sinkhorn_eps: input.hc_sinkhorn_eps,
        hc_post_mult_value: input.hc_post_mult_value,
        residual: input.residual.as_ptr(),
        fn_weights: input.fn_weights.as_ptr(),
        hc_scale: input.hc_scale.as_ptr(),
        hc_base: input.hc_base.as_ptr(),
        post_mix: post_mix.as_mut_ptr(),
        comb_mix: comb_mix.as_mut_ptr(),
        layer_input: layer_input.as_mut_ptr(),
    };
    let mut out = NervaCudaDeepSeekMhcPreResult::default();
    let return_code = run_deepseek_mhc_pre(&request, &mut out);
    summarize(return_code, out, post_mix, comb_mix, layer_input)
}

fn valid_shape(input: &CudaDeepSeekMhcPreInput<'_>) -> bool {
    let Some((post_values, comb_values, layer_values)) = output_values(input) else {
        return false;
    };
    let Some(hc_hidden_values) = (input.hc_mult as usize).checked_mul(input.hidden_size as usize)
    else {
        return false;
    };
    let Some(hc_mult2) = (input.hc_mult as usize).checked_mul(input.hc_mult as usize) else {
        return false;
    };
    let Some(hc_mult3) = (input.hc_mult as usize).checked_mul(2 + input.hc_mult as usize) else {
        return false;
    };
    let Some(residual_values) = (input.tokens as usize).checked_mul(hc_hidden_values) else {
        return false;
    };
    let Some(fn_values) = hc_mult3.checked_mul(hc_hidden_values) else {
        return false;
    };
    input.tokens > 0
        && input.hc_mult > 0
        && input.hc_mult <= MAX_HC_MULT
        && input.hidden_size > 0
        && input.sinkhorn_repeat > 0
        && input.rms_eps.is_finite()
        && input.rms_eps >= 0.0
        && input.hc_pre_eps.is_finite()
        && input.hc_sinkhorn_eps.is_finite()
        && input.hc_post_mult_value.is_finite()
        && hc_hidden_values <= MAX_HC_HIDDEN
        && residual_values <= u32::MAX as usize
        && fn_values <= u32::MAX as usize
        && post_values <= u32::MAX as usize
        && comb_values <= u32::MAX as usize
        && layer_values <= u32::MAX as usize
        && input.residual.len() == residual_values
        && input.fn_weights.len() == fn_values
        && input.hc_scale.len() == 3
        && input.hc_base.len() == hc_mult3
        && hc_mult2 == comb_values / input.tokens as usize
}

fn output_values(input: &CudaDeepSeekMhcPreInput<'_>) -> Option<(usize, usize, usize)> {
    let post = (input.tokens as usize).checked_mul(input.hc_mult as usize)?;
    let comb = post.checked_mul(input.hc_mult as usize)?;
    let layer = (input.tokens as usize).checked_mul(input.hidden_size as usize)?;
    Some((post, comb, layer))
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekMhcPreResult,
    post_mix: Vec<f32>,
    comb_mix: Vec<f32>,
    layer_input: Vec<f32>,
) -> CudaDeepSeekMhcPreSummary {
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
            "CUDA DeepSeek mHC pre failed: return_code={} status={} cuda_error={} mhc_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.mhc_error, out.device_count
        ))
    };
    CudaDeepSeekMhcPreSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        mhc_error: out.mhc_error,
        tokens: out.tokens,
        hc_mult: out.hc_mult,
        hidden_size: out.hidden_size,
        sinkhorn_repeat: out.sinkhorn_repeat,
        rms_eps: out.rms_eps,
        hc_pre_eps: out.hc_pre_eps,
        hc_sinkhorn_eps: out.hc_sinkhorn_eps,
        hc_post_mult_value: out.hc_post_mult_value,
        post_mix,
        comb_mix,
        layer_input,
        post_mix_hash: out.post_mix_hash,
        comb_mix_hash: out.comb_mix_hash,
        layer_input_hash: out.layer_input_hash,
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
    input: &CudaDeepSeekMhcPreInput<'_>,
    post_mix: Vec<f32>,
    comb_mix: Vec<f32>,
    layer_input: Vec<f32>,
    error: impl Into<String>,
) -> CudaDeepSeekMhcPreSummary {
    CudaDeepSeekMhcPreSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        mhc_error: -1,
        tokens: input.tokens,
        hc_mult: input.hc_mult,
        hidden_size: input.hidden_size,
        sinkhorn_repeat: input.sinkhorn_repeat,
        rms_eps: input.rms_eps,
        hc_pre_eps: input.hc_pre_eps,
        hc_sinkhorn_eps: input.hc_sinkhorn_eps,
        hc_post_mult_value: input.hc_post_mult_value,
        post_mix,
        comb_mix,
        layer_input,
        post_mix_hash: 0,
        comb_mix_hash: 0,
        layer_input_hash: 0,
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
