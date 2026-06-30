use crate::deepseek_quant::ffi::{
    NervaCudaDeepSeekFusedInvRopeFp8QuantRequest, NervaCudaDeepSeekFusedInvRopeFp8QuantResult,
    run_deepseek_fused_inv_rope_fp8_quant,
};
use crate::json::json_opt_str;
use crate::smoke::ffi::CUDA_ERROR_NO_DEVICE;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, PartialEq)]
pub struct CudaDeepSeekFusedInvRopeFp8QuantSummary {
    pub status: SmokeStatus,
    pub return_code: i32,
    pub cuda_error: i32,
    pub num_tokens: u32,
    pub n_groups: u32,
    pub heads_per_group: u32,
    pub head_dim: u32,
    pub rope_dim: u32,
    pub quant_group_size: u32,
    pub scale_blocks: u32,
    pub fp8_output: Vec<u8>,
    pub scale_output: Vec<f32>,
    pub packed_scale_output: Vec<u32>,
    pub fp8_output_hash: u64,
    pub scale_output_hash: u64,
    pub packed_scale_output_hash: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaDeepSeekFusedInvRopeFp8QuantSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            SmokeStatus::Ok => "ok",
            SmokeStatus::Unavailable => "unavailable",
            SmokeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"return_code\":{},\"cuda_error\":{},\"num_tokens\":{},\"n_groups\":{},\"heads_per_group\":{},\"head_dim\":{},\"rope_dim\":{},\"quant_group_size\":{},\"scale_blocks\":{},\"fp8_output\":{},\"scale_output\":{},\"packed_scale_output\":{},\"fp8_output_hash\":{},\"scale_output_hash\":{},\"packed_scale_output_hash\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status,
            self.return_code,
            self.cuda_error,
            self.num_tokens,
            self.n_groups,
            self.heads_per_group,
            self.head_dim,
            self.rope_dim,
            self.quant_group_size,
            self.scale_blocks,
            json_u8_array(&self.fp8_output),
            json_f32_array(&self.scale_output),
            json_u32_array(&self.packed_scale_output),
            self.fp8_output_hash,
            self.scale_output_hash,
            self.packed_scale_output_hash,
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

#[allow(clippy::too_many_arguments)]
pub fn deepseek_fused_inv_rope_fp8_quant(
    input: &[f32],
    positions: &[i64],
    cos_sin_cache: &[f32],
    num_tokens: u32,
    n_groups: u32,
    heads_per_group: u32,
    head_dim: u32,
    rope_dim: u32,
    quant_group_size: u32,
    cos_sin_stride: u32,
    fp8_max: f32,
    eps: f32,
) -> CudaDeepSeekFusedInvRopeFp8QuantSummary {
    let chunks_per_head = if quant_group_size == 0 {
        0
    } else {
        head_dim / quant_group_size
    };
    let scale_blocks = heads_per_group.saturating_mul(chunks_per_head);
    let num_heads = n_groups.saturating_mul(heads_per_group);
    let input_values = (num_tokens as usize)
        .checked_mul(num_heads as usize)
        .and_then(|value| value.checked_mul(head_dim as usize))
        .unwrap_or(usize::MAX);
    let fp8_values = (n_groups as usize)
        .checked_mul(num_tokens as usize)
        .and_then(|value| value.checked_mul(heads_per_group as usize))
        .and_then(|value| value.checked_mul(head_dim as usize))
        .unwrap_or(usize::MAX);
    let scale_values = (n_groups as usize)
        .checked_mul(num_tokens as usize)
        .and_then(|value| value.checked_mul(scale_blocks as usize))
        .unwrap_or(usize::MAX);
    let packed_values = (n_groups as usize)
        .checked_mul(num_tokens as usize)
        .and_then(|value| value.checked_mul(heads_per_group as usize))
        .unwrap_or(usize::MAX);
    let max_position = positions
        .iter()
        .copied()
        .filter(|position| *position >= 0)
        .max()
        .unwrap_or(0) as usize;
    let cos_sin_values = (max_position + 1)
        .checked_mul(cos_sin_stride as usize)
        .unwrap_or(usize::MAX);
    if num_tokens == 0
        || n_groups == 0
        || heads_per_group == 0
        || head_dim == 0
        || rope_dim > head_dim
        || rope_dim % 2 != 0
        || quant_group_size == 0
        || head_dim % quant_group_size != 0
        || chunks_per_head == 0
        || chunks_per_head > 4
        || cos_sin_stride < rope_dim
        || !fp8_max.is_finite()
        || !eps.is_finite()
        || fp8_max <= 0.0
        || eps <= 0.0
        || positions.len() != num_tokens as usize
        || input.len() != input_values
        || cos_sin_cache.len() < cos_sin_values
        || input_values == usize::MAX
        || fp8_values == usize::MAX
        || scale_values == usize::MAX
        || packed_values == usize::MAX
        || input_values > u32::MAX as usize
        || fp8_values > u32::MAX as usize
        || scale_values > u32::MAX as usize
        || packed_values > u32::MAX as usize
    {
        return failed_summary(
            num_tokens,
            n_groups,
            heads_per_group,
            head_dim,
            rope_dim,
            quant_group_size,
            scale_blocks,
            "invalid DeepSeek fused inverse RoPE FP8 quant shape",
        );
    }

    let mut fp8_output = vec![0u8; fp8_values];
    let mut scale_output = vec![0.0f32; scale_values];
    let mut packed_scale_output = vec![0u32; packed_values];
    let request = NervaCudaDeepSeekFusedInvRopeFp8QuantRequest {
        num_tokens,
        n_groups,
        heads_per_group,
        head_dim,
        rope_dim,
        quant_group_size,
        cos_sin_stride,
        fp8_max,
        eps,
        input: input.as_ptr(),
        positions: positions.as_ptr(),
        cos_sin_cache: cos_sin_cache.as_ptr(),
        fp8_output: fp8_output.as_mut_ptr(),
        scale_output: scale_output.as_mut_ptr(),
        packed_scale_output: packed_scale_output.as_mut_ptr(),
    };
    let mut out = NervaCudaDeepSeekFusedInvRopeFp8QuantResult::default();
    let return_code = run_deepseek_fused_inv_rope_fp8_quant(&request, &mut out);
    summarize(
        return_code,
        out,
        fp8_output,
        scale_output,
        packed_scale_output,
    )
}

fn summarize(
    return_code: i32,
    out: NervaCudaDeepSeekFusedInvRopeFp8QuantResult,
    fp8_output: Vec<u8>,
    scale_output: Vec<f32>,
    packed_scale_output: Vec<u32>,
) -> CudaDeepSeekFusedInvRopeFp8QuantSummary {
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
            "CUDA DeepSeek fused inverse RoPE FP8 quant failed: return_code={} status={} cuda_error={} device_count={}",
            return_code, out.status, out.cuda_error, out.device_count
        ))
    };
    CudaDeepSeekFusedInvRopeFp8QuantSummary {
        status,
        return_code,
        cuda_error: out.cuda_error,
        num_tokens: out.num_tokens,
        n_groups: out.n_groups,
        heads_per_group: out.heads_per_group,
        head_dim: out.head_dim,
        rope_dim: out.rope_dim,
        quant_group_size: out.quant_group_size,
        scale_blocks: out.scale_blocks,
        fp8_output,
        scale_output,
        packed_scale_output,
        fp8_output_hash: out.fp8_output_hash,
        scale_output_hash: out.scale_output_hash,
        packed_scale_output_hash: out.packed_scale_output_hash,
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
    n_groups: u32,
    heads_per_group: u32,
    head_dim: u32,
    rope_dim: u32,
    quant_group_size: u32,
    scale_blocks: u32,
    error: impl Into<String>,
) -> CudaDeepSeekFusedInvRopeFp8QuantSummary {
    CudaDeepSeekFusedInvRopeFp8QuantSummary {
        status: SmokeStatus::Failed,
        return_code: -1,
        cuda_error: 0,
        num_tokens,
        n_groups,
        heads_per_group,
        head_dim,
        rope_dim,
        quant_group_size,
        scale_blocks,
        fp8_output: Vec::new(),
        scale_output: Vec::new(),
        packed_scale_output: Vec::new(),
        fp8_output_hash: 0,
        scale_output_hash: 0,
        packed_scale_output_hash: 0,
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

fn json_u8_array(values: &[u8]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}

fn json_u32_array(values: &[u32]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
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
