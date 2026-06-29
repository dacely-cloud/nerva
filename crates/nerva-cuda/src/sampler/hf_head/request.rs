use crate::sampler::ffi::{
    CUDA_ERROR_NO_DEVICE, NervaCudaHfSamplerRequest, NervaCudaHfSamplerResult, run_hf_sample_u16,
};
use crate::sampler::hf_head::summary::{CudaHfSamplerSummary, empty_summary};
use crate::smoke::status::SmokeStatus;

pub const CUDA_HF_SAMPLER_DTYPE_F16: u32 = 0;
pub const CUDA_HF_SAMPLER_DTYPE_BF16: u32 = 1;

#[derive(Clone, Debug)]
pub struct CudaHfSamplerRequest<'a> {
    pub dtype: u32,
    pub hidden: usize,
    pub vocab_size: usize,
    pub token_index: u64,
    pub rms_eps: f32,
    pub hidden_bits: &'a [u16],
    pub final_norm_weight: &'a [u16],
    pub lm_head: &'a [u16],
}

impl<'a> CudaHfSamplerRequest<'a> {
    pub fn run(&self) -> CudaHfSamplerSummary {
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
        let mut out = NervaCudaHfSamplerResult::default();
        let return_code = run_hf_sample_u16(&ffi_request, &mut out);
        let status = if return_code == 0 && out.status == 0 {
            SmokeStatus::Ok
        } else if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
            SmokeStatus::Unavailable
        } else {
            SmokeStatus::Failed
        };
        let error = (status != SmokeStatus::Ok).then(|| failure_reason(return_code, &out));
        CudaHfSamplerSummary {
            status,
            dtype: out.dtype,
            hidden: out.hidden,
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
        if self.hidden == 0 || self.vocab_size == 0 {
            return Some("CUDA HF sampler dimensions must be non-zero".to_string());
        }
        if self.dtype > CUDA_HF_SAMPLER_DTYPE_BF16 {
            return Some("CUDA HF sampler dtype is unsupported".to_string());
        }
        if self.hidden_bits.len() != self.hidden {
            return Some("CUDA HF sampler hidden width mismatch".to_string());
        }
        if self.final_norm_weight.len() != self.hidden {
            return Some("CUDA HF sampler final norm width mismatch".to_string());
        }
        if self.lm_head.len() != self.hidden * self.vocab_size {
            return Some("CUDA HF sampler LM head shape mismatch".to_string());
        }
        None
    }

    fn to_ffi(&self) -> NervaCudaHfSamplerRequest {
        NervaCudaHfSamplerRequest {
            dtype: self.dtype,
            hidden: self.hidden as u32,
            vocab_size: self.vocab_size as u32,
            token_index: self.token_index,
            rms_eps: self.rms_eps,
            hidden_bits: self.hidden_bits.as_ptr(),
            final_norm_weight: self.final_norm_weight.as_ptr(),
            lm_head: self.lm_head.as_ptr(),
        }
    }
}

fn failure_reason(return_code: i32, out: &NervaCudaHfSamplerResult) -> String {
    format!(
        "CUDA HF sampler failed: return_code={} status={} cuda_error={} device_count={} hidden={} vocab={} token={}",
        return_code,
        out.status,
        out.cuda_error,
        out.device_count,
        out.hidden,
        out.vocab_size,
        out.token,
    )
}
