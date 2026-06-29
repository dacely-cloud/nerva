use crate::decode::ffi::CUDA_ERROR_NO_DEVICE;
use crate::decode::hf_chain::ffi::{
    run_hf_decode_chain_u16, NervaCudaHfDecodeChainRequest, NervaCudaHfDecodeChainResult,
};
use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_chain::summary::{empty_summary, CudaHfDecodeChainSummary};
use crate::smoke::status::SmokeStatus;

pub const CUDA_HF_DECODE_CHAIN_DTYPE_F16: u32 = 0;
pub const CUDA_HF_DECODE_CHAIN_DTYPE_BF16: u32 = 1;

#[derive(Clone, Debug)]
pub struct CudaHfDecodeChainRequest<'a> {
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
    pub layers: &'a [CudaHfDecodeChainLayer<'a>],
    pub final_norm_weight: &'a [u16],
    pub lm_head: &'a [u16],
}

impl<'a> CudaHfDecodeChainRequest<'a> {
    pub fn run(&self) -> CudaHfDecodeChainSummary {
        if let Some(error) = self.validate() {
            return empty_summary(
                SmokeStatus::Failed,
                self.dtype,
                self.hidden,
                self.vocab_size,
                self.layers.len(),
                self.token_index,
                error,
            );
        }
        let ffi_layers = self
            .layers
            .iter()
            .map(|layer| layer.to_ffi())
            .collect::<Vec<_>>();
        let ffi_request = self.to_ffi(ffi_layers.as_ptr());
        let mut out = NervaCudaHfDecodeChainResult::default();
        let return_code = run_hf_decode_chain_u16(&ffi_request, &mut out);
        let status = status_from_result(return_code, &out);
        let error = (status != SmokeStatus::Ok).then(|| failure_reason(return_code, &out));
        CudaHfDecodeChainSummary {
            status,
            dtype: out.dtype,
            hidden: out.hidden,
            heads: out.heads,
            kv_heads: out.kv_heads,
            head_dim: out.head_dim,
            intermediate: out.intermediate,
            vocab_size: out.vocab_size,
            layer_count: out.layer_count,
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
            return Some("CUDA HF decode chain dimensions must be non-zero".to_string());
        }
        if self.vocab_size == 0 || self.intermediate == 0 || self.layers.is_empty() {
            return Some("CUDA HF decode chain layers and vocabulary must be non-zero".to_string());
        }
        if self.kv_heads > self.heads || !self.heads.is_multiple_of(self.kv_heads) {
            return Some("CUDA HF decode chain KV heads must divide attention heads".to_string());
        }
        if self.dtype > CUDA_HF_DECODE_CHAIN_DTYPE_BF16 {
            return Some("CUDA HF decode chain dtype is unsupported".to_string());
        }
        if self.rope_theta.is_some() && !self.head_dim.is_multiple_of(2) {
            return Some("CUDA HF decode chain RoPE requires an even head dimension".to_string());
        }
        self.validate_lengths()
    }

    fn validate_lengths(&self) -> Option<String> {
        if self.input.len() != self.hidden {
            return Some("CUDA HF decode chain input length does not match hidden".to_string());
        }
        if self.final_norm_weight.len() != self.hidden {
            return Some(
                "CUDA HF decode chain final norm length does not match hidden".to_string(),
            );
        }
        if self.lm_head.len() != self.vocab_size * self.hidden {
            return Some("CUDA HF decode chain LM head length does not match shape".to_string());
        }
        let attention_hidden = self.heads * self.head_dim;
        let kv_hidden = self.kv_heads * self.head_dim;
        self.layers.iter().enumerate().find_map(|(index, layer)| {
            layer
                .validate(
                    self.hidden,
                    attention_hidden,
                    kv_hidden,
                    self.head_dim,
                    self.intermediate,
                )
                .map(|error| format!("layer {index}: {error}"))
        })
    }

    fn to_ffi(
        &self,
        layers: *const crate::decode::hf_chain::ffi::NervaCudaHfDecodeChainLayer,
    ) -> NervaCudaHfDecodeChainRequest {
        NervaCudaHfDecodeChainRequest {
            dtype: self.dtype,
            hidden: self.hidden as u32,
            heads: self.heads as u32,
            kv_heads: self.kv_heads as u32,
            head_dim: self.head_dim as u32,
            intermediate: self.intermediate as u32,
            vocab_size: self.vocab_size as u32,
            layer_count: self.layers.len() as u32,
            position: self.position,
            token_index: self.token_index,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta.unwrap_or(0.0),
            input: self.input.as_ptr(),
            layers,
            final_norm_weight: self.final_norm_weight.as_ptr(),
            lm_head: self.lm_head.as_ptr(),
        }
    }
}

fn status_from_result(return_code: i32, out: &NervaCudaHfDecodeChainResult) -> SmokeStatus {
    if return_code == 0 && out.status == 0 {
        SmokeStatus::Ok
    } else if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        SmokeStatus::Unavailable
    } else {
        SmokeStatus::Failed
    }
}

fn failure_reason(return_code: i32, out: &NervaCudaHfDecodeChainResult) -> String {
    format!(
        "CUDA HF decode chain failed: return_code={return_code} status={} cuda_error={}",
        out.status, out.cuda_error,
    )
}
