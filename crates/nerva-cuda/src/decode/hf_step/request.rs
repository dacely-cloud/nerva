use core::ptr;

use crate::decode::ffi::{
    CUDA_ERROR_NO_DEVICE, NervaCudaHfDecodeStepRequest, NervaCudaHfDecodeStepResult,
    run_hf_decode_step_u16,
};
use crate::decode::hf_step::summary::{CudaHfDecodeStepSummary, empty_summary};
use crate::smoke::status::SmokeStatus;
pub const CUDA_HF_DECODE_STEP_DTYPE_F16: u32 = 0;
pub const CUDA_HF_DECODE_STEP_DTYPE_BF16: u32 = 1;

#[derive(Clone, Debug)]
pub struct CudaHfDecodeStepRequest<'a> {
    pub dtype: u32,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab_size: usize,
    pub position: u32,
    pub token_index: u64,
    pub rms_eps: f32,
    pub rope_theta: Option<f32>,
    pub input: &'a [u16],
    pub rms_attn_weight: &'a [u16],
    pub rms_mlp_weight: &'a [u16],
    pub w_q: &'a [u16],
    pub w_k: &'a [u16],
    pub w_v: &'a [u16],
    pub w_o: &'a [u16],
    pub q_bias: Option<&'a [u16]>,
    pub k_bias: Option<&'a [u16]>,
    pub v_bias: Option<&'a [u16]>,
    pub o_bias: Option<&'a [u16]>,
    pub w_gate: &'a [u16],
    pub w_up: &'a [u16],
    pub w_down: &'a [u16],
    pub final_norm_weight: &'a [u16],
    pub lm_head: &'a [u16],
}
impl<'a> CudaHfDecodeStepRequest<'a> {
    pub fn run(&self) -> CudaHfDecodeStepSummary {
        if let Some(error) = self.validate() {
            return empty_summary(
                SmokeStatus::Failed,
                self.dtype,
                self.hidden,
                self.vocab_size,
                self.token_index,
                error,
            );
        }
        let ffi_request = self.to_ffi();
        let mut out = NervaCudaHfDecodeStepResult::default();
        let return_code = run_hf_decode_step_u16(&ffi_request, &mut out);
        let status = if return_code == 0 && out.status == 0 {
            SmokeStatus::Ok
        } else if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
            SmokeStatus::Unavailable
        } else {
            SmokeStatus::Failed
        };
        let error = (status != SmokeStatus::Ok).then(|| failure_reason(return_code, &out));
        CudaHfDecodeStepSummary {
            status,
            dtype: out.dtype,
            hidden: out.hidden,
            heads: out.heads,
            kv_heads: out.kv_heads,
            head_dim: out.head_dim,
            intermediate: out.intermediate,
            vocab_size: out.vocab_size,
            token_index: out.token_index,
            token: out.token,
            slot_version: out.slot_version,
            completion: out.completion,
            output_hash: out.output_hash,
            resident_weight_bytes: out.resident_weight_bytes,
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

    fn validate(&self) -> Option<String> {
        if self.hidden == 0 || self.heads == 0 || self.kv_heads == 0 || self.head_dim == 0 {
            return Some("CUDA HF decode step dimensions must be non-zero".to_string());
        }
        if self.vocab_size == 0 || self.intermediate == 0 {
            return Some("CUDA HF decode step MLP and vocabulary must be non-zero".to_string());
        }
        if self.kv_heads > self.heads || !self.heads.is_multiple_of(self.kv_heads) {
            return Some("CUDA HF decode step KV heads must divide attention heads".to_string());
        }
        if self.dtype > CUDA_HF_DECODE_STEP_DTYPE_BF16 {
            return Some("CUDA HF decode step dtype is unsupported".to_string());
        }
        if self.rope_theta.is_some() && !self.head_dim.is_multiple_of(2) {
            return Some("CUDA HF decode step RoPE requires an even head dimension".to_string());
        }
        self.validate_lengths()
    }

    fn validate_lengths(&self) -> Option<String> {
        let attention_hidden = self.heads * self.head_dim;
        let kv_hidden = self.kv_heads * self.head_dim;
        for (name, actual, expected) in self.required_lengths(attention_hidden, kv_hidden) {
            if actual != expected {
                return Some(format!(
                    "CUDA HF decode step {name} length {actual} != {expected}"
                ));
            }
        }
        validate_optional("q_bias", self.q_bias, attention_hidden)
            .or_else(|| validate_optional("k_bias", self.k_bias, kv_hidden))
            .or_else(|| validate_optional("v_bias", self.v_bias, kv_hidden))
            .or_else(|| validate_optional("o_bias", self.o_bias, self.hidden))
    }

    fn required_lengths(
        &self,
        attention_hidden: usize,
        kv_hidden: usize,
    ) -> [(&'static str, usize, usize); 12] {
        [
            ("input", self.input.len(), self.hidden),
            ("rms_attn_weight", self.rms_attn_weight.len(), self.hidden),
            ("rms_mlp_weight", self.rms_mlp_weight.len(), self.hidden),
            ("w_q", self.w_q.len(), attention_hidden * self.hidden),
            ("w_k", self.w_k.len(), kv_hidden * self.hidden),
            ("w_v", self.w_v.len(), kv_hidden * self.hidden),
            ("w_o", self.w_o.len(), self.hidden * attention_hidden),
            ("w_gate", self.w_gate.len(), self.intermediate * self.hidden),
            ("w_up", self.w_up.len(), self.intermediate * self.hidden),
            ("w_down", self.w_down.len(), self.hidden * self.intermediate),
            ("final_norm", self.final_norm_weight.len(), self.hidden),
            ("lm_head", self.lm_head.len(), self.vocab_size * self.hidden),
        ]
    }

    fn to_ffi(&self) -> NervaCudaHfDecodeStepRequest {
        NervaCudaHfDecodeStepRequest {
            dtype: self.dtype,
            hidden: self.hidden as u32,
            heads: self.heads as u32,
            kv_heads: self.kv_heads as u32,
            head_dim: self.head_dim as u32,
            intermediate: self.intermediate as u32,
            vocab_size: self.vocab_size as u32,
            position: self.position,
            token_index: self.token_index,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta.unwrap_or(0.0),
            input: self.input.as_ptr(),
            rms_attn_weight: self.rms_attn_weight.as_ptr(),
            rms_mlp_weight: self.rms_mlp_weight.as_ptr(),
            w_q: self.w_q.as_ptr(),
            w_k: self.w_k.as_ptr(),
            w_v: self.w_v.as_ptr(),
            w_o: self.w_o.as_ptr(),
            q_bias: optional_ptr(self.q_bias),
            k_bias: optional_ptr(self.k_bias),
            v_bias: optional_ptr(self.v_bias),
            o_bias: optional_ptr(self.o_bias),
            w_gate: self.w_gate.as_ptr(),
            w_up: self.w_up.as_ptr(),
            w_down: self.w_down.as_ptr(),
            final_norm_weight: self.final_norm_weight.as_ptr(),
            lm_head: self.lm_head.as_ptr(),
        }
    }
}

fn validate_optional(name: &'static str, value: Option<&[u16]>, expected: usize) -> Option<String> {
    match value {
        Some(slice) if slice.len() != expected => Some(format!(
            "CUDA HF decode step {name} length {} != {expected}",
            slice.len()
        )),
        _ => None,
    }
}

fn optional_ptr(slice: Option<&[u16]>) -> *const u16 {
    slice.map_or(ptr::null(), <[u16]>::as_ptr)
}

fn failure_reason(return_code: i32, out: &NervaCudaHfDecodeStepResult) -> String {
    format!(
        "CUDA HF decode step failed: return_code={return_code} status={} cuda_error={}",
        out.status, out.cuda_error,
    )
}
