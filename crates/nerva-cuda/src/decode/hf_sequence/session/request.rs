use core::ptr;

use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::ffi::NervaCudaHfDecodeSequenceResult;
use crate::decode::hf_sequence::request::CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16;
use crate::decode::hf_sequence::session::failures::{failed_create_summary, failed_run_summary};
use crate::decode::hf_sequence::session::ffi::{
    NervaCudaHfDecodeSequenceSession, NervaCudaHfDecodeSequenceSessionCreateRequest,
    NervaCudaHfDecodeSequenceSessionCreateResult, NervaCudaHfDecodeSequenceSessionRunRequest,
    create_hf_decode_sequence_session, destroy_hf_decode_sequence_session,
    run_hf_decode_sequence_session,
};
use crate::decode::hf_sequence::session::helpers::{
    descriptor_ptr, planned_ptr, summary_from_run, validate_run,
};
use crate::decode::hf_sequence::session::summary::{
    CudaHfDecodeSequenceSessionCreateSummary, create_summary_from_result,
};
use crate::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use crate::decode::hf_sequence::weight_plan::{
    CudaHfDecodeSequenceWeightBlock, CudaHfDecodeSequenceWeightPlan,
};
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug)]
pub struct CudaHfDecodeSequenceSessionConfig<'a> {
    pub dtype: u32,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab_size: usize,
    pub max_context_tokens: usize,
    pub rms_eps: f32,
    pub rope_theta: Option<f32>,
    pub embeddings: &'a [u16],
    pub layers: &'a [CudaHfDecodeChainLayer<'a>],
    pub final_norm_weight: &'a [u16],
    pub lm_head: &'a [u16],
    pub weight_plan: Option<CudaHfDecodeSequenceWeightPlan>,
    pub weight_blocks: &'a [CudaHfDecodeSequenceWeightBlock],
    pub detailed_profile: bool,
}

pub struct CudaHfDecodeSequenceSession {
    handle: *mut NervaCudaHfDecodeSequenceSession,
    create_summary: CudaHfDecodeSequenceSessionCreateSummary,
}

pub struct CudaHfDecodeSequenceSessionCreateOutput {
    pub summary: CudaHfDecodeSequenceSessionCreateSummary,
    pub session: Option<CudaHfDecodeSequenceSession>,
}

impl<'a> CudaHfDecodeSequenceSessionConfig<'a> {
    pub fn create(&self) -> CudaHfDecodeSequenceSessionCreateOutput {
        if let Some(error) = validate_config(self) {
            return CudaHfDecodeSequenceSessionCreateOutput {
                summary: failed_create_summary(self, error),
                session: None,
            };
        }
        let plan = self.weight_plan.unwrap_or_default();
        let use_descriptors = plan.is_declared();
        let ffi_layers = self
            .layers
            .iter()
            .map(|layer| {
                if use_descriptors {
                    layer.to_descriptor_layout_ffi()
                } else {
                    layer.to_ffi()
                }
            })
            .collect::<Vec<_>>();
        let mut handle = ptr::null_mut();
        let request = self.to_ffi(ffi_layers.as_ptr());
        let mut out = NervaCudaHfDecodeSequenceSessionCreateResult::default();
        let admission_memory = crate::smoke::probe::smoke();
        let return_code = create_hf_decode_sequence_session(&request, &mut out, &mut handle);
        let summary = create_summary_from_result(
            return_code,
            &out,
            admission_memory.device_total_memory_bytes,
            admission_memory.device_free_memory_bytes,
        );
        let session = (summary.status == SmokeStatus::Ok && !handle.is_null()).then(|| {
            CudaHfDecodeSequenceSession {
                handle,
                create_summary: summary.clone(),
            }
        });
        CudaHfDecodeSequenceSessionCreateOutput { summary, session }
    }

    fn to_ffi(
        &self,
        layers: *const crate::decode::hf_chain::ffi::NervaCudaHfDecodeChainLayer,
    ) -> NervaCudaHfDecodeSequenceSessionCreateRequest {
        let plan = self.weight_plan.unwrap_or_default();
        NervaCudaHfDecodeSequenceSessionCreateRequest {
            dtype: self.dtype,
            hidden: self.hidden as u32,
            heads: self.heads as u32,
            kv_heads: self.kv_heads as u32,
            head_dim: self.head_dim as u32,
            intermediate: self.intermediate as u32,
            vocab_size: self.vocab_size as u32,
            layer_count: self.layers.len() as u32,
            max_context_tokens: self.max_context_tokens as u32,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta.unwrap_or(0.0),
            embeddings: planned_ptr(self.embeddings, plan),
            layers,
            final_norm_weight: planned_ptr(self.final_norm_weight, plan),
            lm_head: planned_ptr(self.lm_head, plan),
            planned_weight_blocks: plan.blocks,
            planned_gpu_resident_blocks: plan.gpu_resident_blocks,
            planned_gpu_staged_blocks: plan.gpu_staged_blocks,
            planned_weight_bytes: plan.weight_bytes,
            planned_gpu_resident_weight_bytes: plan.gpu_resident_weight_bytes,
            planned_gpu_staged_weight_bytes: plan.gpu_staged_weight_bytes,
            planned_weight_descriptors: descriptor_ptr(self.weight_blocks),
            planned_weight_descriptor_count: self.weight_blocks.len() as u32,
            planned_weight_descriptor_hash: plan.descriptor_hash,
            detailed_profile: self.detailed_profile as u32,
        }
    }
}

impl CudaHfDecodeSequenceSession {
    pub fn create_summary(&self) -> &CudaHfDecodeSequenceSessionCreateSummary {
        &self.create_summary
    }

    pub(super) fn raw_handle(&mut self) -> *mut NervaCudaHfDecodeSequenceSession {
        self.handle
    }

    pub fn run(
        &mut self,
        prompt_tokens: &[u32],
        steps: usize,
        eos_token: Option<u32>,
    ) -> CudaHfDecodeSequenceSummary {
        if let Some(error) = validate_run(prompt_tokens, steps, self.create_summary.vocab_size) {
            return failed_run_summary(&self.create_summary, steps, 0, error);
        }
        let mut tokens = vec![0u32; steps];
        let seed = *prompt_tokens.last().unwrap();
        let request = NervaCudaHfDecodeSequenceSessionRunRequest {
            session: self.handle,
            steps: steps as u32,
            seed_token: seed,
            prompt_tokens: prompt_tokens.as_ptr(),
            prompt_token_count: prompt_tokens.len() as u32,
            has_eos_token: eos_token.is_some() as u32,
            eos_token: eos_token.unwrap_or(0),
            output_tokens: tokens.as_mut_ptr(),
            output_token_capacity: steps as u32,
        };
        let mut out = NervaCudaHfDecodeSequenceResult::default();
        let return_code = run_hf_decode_sequence_session(&request, &mut out);
        tokens.truncate(out.observed_tokens.min(steps as u32) as usize);
        summary_from_run(return_code, &out, tokens, &self.create_summary)
    }
}

impl Drop for CudaHfDecodeSequenceSession {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let mut out = NervaCudaHfDecodeSequenceSessionCreateResult::default();
            let _ = destroy_hf_decode_sequence_session(self.handle, &mut out);
            self.handle = ptr::null_mut();
        }
    }
}

fn validate_config(request: &CudaHfDecodeSequenceSessionConfig<'_>) -> Option<String> {
    if request.hidden == 0 || request.heads == 0 || request.kv_heads == 0 || request.head_dim == 0 {
        return Some("CUDA HF decode sequence session dimensions must be non-zero".to_string());
    }
    if request.layers.is_empty() || request.max_context_tokens == 0 {
        return Some("CUDA HF decode sequence session requires layers and capacity".to_string());
    }
    if request.dtype > CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16 {
        return Some("CUDA HF decode sequence session dtype is unsupported".to_string());
    }
    request.weight_plan.and_then(|plan| {
        plan.validate()
            .or_else(|| plan.validate_descriptors(request.weight_blocks))
    })
}
