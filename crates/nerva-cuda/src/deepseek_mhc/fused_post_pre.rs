use crate::deepseek_mhc::ffi::{
    NervaCudaDeepSeekMhcFusedPostPreRequest, NervaCudaDeepSeekMhcFusedPostPreResult,
    run_deepseek_mhc_fused_post_pre,
};
use crate::json::json_opt_str;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

const MAX_HC_MULT: u32 = 8;
const MAX_HC_HIDDEN: usize = 65_536;

#[derive(Clone, Debug)]
pub struct CudaDeepSeekMhcFusedPostPreInput<'a> {
    pub tokens: u32,
    pub hc_mult: u32,
    pub hidden_size: u32,
    pub sinkhorn_repeat: u32,
    pub rms_eps: f32,
    pub hc_pre_eps: f32,
    pub hc_sinkhorn_eps: f32,
    pub hc_post_mult_value: f32,
    pub x: &'a [f32],
    pub residual: &'a [f32],
    pub post_layer_mix: &'a [f32],
    pub comb_res_mix: &'a [f32],
    pub fn_weights: &'a [f32],
    pub hc_scale: &'a [f32],
    pub hc_base: &'a [f32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekMhcFusedPostPreSummary {
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
    pub new_residual: Vec<f32>,
    pub new_post_mix: Vec<f32>,
    pub new_comb_mix: Vec<f32>,
    pub layer_input: Vec<f32>,
    pub new_residual_hash: u64,
    pub new_post_mix_hash: u64,
    pub new_comb_mix_hash: u64,
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

impl CudaDeepSeekMhcFusedPostPreSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"return_code\":{},\"cuda_error\":{},\"mhc_error\":{},\"tokens\":{},\"hc_mult\":{},\"hidden_size\":{},\"sinkhorn_repeat\":{},\"rms_eps\":{},\"hc_pre_eps\":{},\"hc_sinkhorn_eps\":{},\"hc_post_mult_value\":{},\"new_residual\":{},\"new_post_mix\":{},\"new_comb_mix\":{},\"layer_input\":{},\"new_residual_hash\":{},\"new_post_mix_hash\":{},\"new_comb_mix_hash\":{},\"layer_input_hash\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
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
            json_f32_array(&self.new_residual),
            json_f32_array(&self.new_post_mix),
            json_f32_array(&self.new_comb_mix),
            json_f32_array(&self.layer_input),
            self.new_residual_hash,
            self.new_post_mix_hash,
            self.new_comb_mix_hash,
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

pub fn deepseek_mhc_fused_post_pre(
    input: CudaDeepSeekMhcFusedPostPreInput<'_>,
) -> CudaDeepSeekMhcFusedPostPreSummary {
    let output_values = output_values(&input)
        .filter(|(residual, post, comb, layer)| {
            *residual <= u32::MAX as usize
                && *post <= u32::MAX as usize
                && *comb <= u32::MAX as usize
                && *layer <= u32::MAX as usize
        })
        .unwrap_or((0, 0, 0, 0));
    if !valid_shape(&input) {
        return failed_summary(
            &input,
            vec![0.0; output_values.0],
            vec![0.0; output_values.1],
            vec![0.0; output_values.2],
            vec![0.0; output_values.3],
            "invalid DeepSeek mHC fused post-pre shape",
        );
    }

    let mut new_residual = vec![0.0f32; output_values.0];
    let mut new_post_mix = vec![0.0f32; output_values.1];
    let mut new_comb_mix = vec![0.0f32; output_values.2];
    let mut layer_input = vec![0.0f32; output_values.3];
    let request = NervaCudaDeepSeekMhcFusedPostPreRequest {
        tokens: input.tokens,
        hc_mult: input.hc_mult,
        hidden_size: input.hidden_size,
        sinkhorn_repeat: input.sinkhorn_repeat,
        rms_eps: input.rms_eps,
        hc_pre_eps: input.hc_pre_eps,
        hc_sinkhorn_eps: input.hc_sinkhorn_eps,
        hc_post_mult_value: input.hc_post_mult_value,
        x: input.x.as_ptr(),
        residual: input.residual.as_ptr(),
        post_layer_mix: input.post_layer_mix.as_ptr(),
        comb_res_mix: input.comb_res_mix.as_ptr(),
        fn_weights: input.fn_weights.as_ptr(),
        hc_scale: input.hc_scale.as_ptr(),
        hc_base: input.hc_base.as_ptr(),
        new_residual: new_residual.as_mut_ptr(),
        new_post_mix: new_post_mix.as_mut_ptr(),
        new_comb_mix: new_comb_mix.as_mut_ptr(),
        layer_input: layer_input.as_mut_ptr(),
    };
    let mut out = NervaCudaDeepSeekMhcFusedPostPreResult::default();
    let return_code = run_deepseek_mhc_fused_post_pre(&request, &mut out);
    summarize(
        return_code,
        out,
        new_residual,
        new_post_mix,
        new_comb_mix,
        layer_input,
    )
}

fn valid_shape(input: &CudaDeepSeekMhcFusedPostPreInput<'_>) -> bool {
    let Some((residual_values, post_values, comb_values, layer_values)) = output_values(input)
    else {
        return false;
    };
    let Some(hc_hidden_values) = (input.hc_mult as usize).checked_mul(input.hidden_size as usize)
    else {
        return false;
    };
    let Some(hc_mult3) =
        (input.hc_mult as usize).checked_mul(2usize.saturating_add(input.hc_mult as usize))
    else {
        return false;
    };
    let Some(fn_values) = hc_mult3.checked_mul(hc_hidden_values) else {
        return false;
    };
    input.tokens != 0
        && input.hc_mult != 0
        && input.hc_mult <= MAX_HC_MULT
        && input.hidden_size != 0
        && input.sinkhorn_repeat != 0
        && hc_hidden_values <= MAX_HC_HIDDEN
        && input.rms_eps.is_finite()
        && input.hc_pre_eps.is_finite()
        && input.hc_sinkhorn_eps.is_finite()
        && input.hc_post_mult_value.is_finite()
        && input.x.len() == layer_values
        && input.residual.len() == residual_values
        && input.post_layer_mix.len() == post_values
        && input.comb_res_mix.len() == comb_values
        && input.fn_weights.len() == fn_values
        && input.hc_scale.len() == 3
        && input.hc_base.len() == hc_mult3
}

fn output_values(
    input: &CudaDeepSeekMhcFusedPostPreInput<'_>,
) -> Option<(usize, usize, usize, usize)> {
    let tokens = input.tokens as usize;
    let hc_mult = input.hc_mult as usize;
    let hidden_size = input.hidden_size as usize;
    let hc_hidden = hc_mult.checked_mul(hidden_size)?;
    let residual = tokens.checked_mul(hc_hidden)?;
    let post = tokens.checked_mul(hc_mult)?;
    let comb = post.checked_mul(hc_mult)?;
    let layer = tokens.checked_mul(hidden_size)?;
    Some((residual, post, comb, layer))
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekMhcFusedPostPreResult,
    new_residual: Vec<f32>,
    new_post_mix: Vec<f32>,
    new_comb_mix: Vec<f32>,
    layer_input: Vec<f32>,
) -> CudaDeepSeekMhcFusedPostPreSummary {
    let status = if return_code == 0 && out.status == 0 {
        SmokeStatus::Ok
    } else if out.cuda_error == CUDA_ERROR_NO_DEVICE {
        SmokeStatus::Unavailable
    } else {
        SmokeStatus::Failed
    };
    let error = (status == SmokeStatus::Failed).then(|| {
        format!(
            "CUDA DeepSeek mHC fused post-pre failed: return_code={} cuda_error={} mhc_error={}",
            return_code, out.cuda_error, out.mhc_error
        )
    });
    CudaDeepSeekMhcFusedPostPreSummary {
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
        new_residual,
        new_post_mix,
        new_comb_mix,
        layer_input,
        new_residual_hash: out.new_residual_hash,
        new_post_mix_hash: out.new_post_mix_hash,
        new_comb_mix_hash: out.new_comb_mix_hash,
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
    input: &CudaDeepSeekMhcFusedPostPreInput<'_>,
    new_residual: Vec<f32>,
    new_post_mix: Vec<f32>,
    new_comb_mix: Vec<f32>,
    layer_input: Vec<f32>,
    error: &str,
) -> CudaDeepSeekMhcFusedPostPreSummary {
    CudaDeepSeekMhcFusedPostPreSummary {
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
        new_residual,
        new_post_mix,
        new_comb_mix,
        layer_input,
        new_residual_hash: 0,
        new_post_mix_hash: 0,
        new_comb_mix_hash: 0,
        layer_input_hash: 0,
        device_arena_bytes: 0,
        pinned_host_bytes: 0,
        h2d_bytes: 0,
        d2h_bytes: 0,
        kernel_launches: 0,
        sync_calls: 0,
        hot_path_allocations: 0,
        error: Some(error.to_string()),
    }
}

fn json_f32_array(values: &[f32]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!("{value:.6}"));
    }
    out.push(']');
    out
}
