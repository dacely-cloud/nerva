use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::ffi::{
    NervaCudaHfDecodeSamplerConfig as FfiSamplerConfig, NervaCudaHfDecodeSequenceRequest,
    NervaCudaHfDecodeSequenceResult, run_hf_decode_sequence_u16,
};
use crate::decode::hf_sequence::footprint::estimate_sequence_footprint;
use crate::decode::hf_sequence::status::{sequence_failure_reason, sequence_status_from_result};
use crate::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use crate::decode::hf_sequence::summary_empty::empty_summary;
use crate::decode::hf_sequence::validation::validate_request;
use crate::decode::hf_sequence::weight_plan::{
    CudaHfDecodeSequenceWeightBlock, CudaHfDecodeSequenceWeightPlan,
};
use crate::smoke::status::SmokeStatus;
pub const CUDA_HF_DECODE_SEQUENCE_DTYPE_F16: u32 = 0;
pub const CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaHfDecodeSamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub seed: u64,
}

impl CudaHfDecodeSamplerConfig {
    pub const fn accuracy_default() -> Self {
        Self::greedy()
    }

    pub const fn vllm_default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
        }
    }

    pub const fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
        }
    }

    pub fn validate(self) -> Option<String> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Some("CUDA HF decode sampler temperature must be finite and >= 0".to_string());
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Some("CUDA HF decode sampler top_p must be finite and in (0, 1]".to_string());
        }
        None
    }

    pub(crate) fn to_ffi(self) -> FfiSamplerConfig {
        FfiSamplerConfig {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            reserved: 0,
            seed: self.seed,
        }
    }
}

impl Default for CudaHfDecodeSamplerConfig {
    fn default() -> Self {
        Self::accuracy_default()
    }
}

#[derive(Clone, Debug)]
pub struct CudaHfDecodeSequenceRequest<'a> {
    pub dtype: u32,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab_size: usize,
    pub steps: usize,
    pub seed_token: u32,
    pub prompt_tokens: &'a [u32],
    pub eos_token: Option<u32>,
    pub rms_eps: f32,
    pub rope_theta: Option<f32>,
    pub embeddings: &'a [u16],
    pub layers: &'a [CudaHfDecodeChainLayer<'a>],
    pub final_norm_weight: &'a [u16],
    pub lm_head: &'a [u16],
    pub weight_plan: Option<CudaHfDecodeSequenceWeightPlan>,
    pub weight_blocks: &'a [CudaHfDecodeSequenceWeightBlock],
    pub sampler: CudaHfDecodeSamplerConfig,
}
impl<'a> CudaHfDecodeSequenceRequest<'a> {
    pub fn run(&self) -> CudaHfDecodeSequenceSummary {
        if let Some(error) = validate_request(self) {
            return empty_summary(
                SmokeStatus::Failed,
                self.dtype,
                self.hidden,
                self.vocab_size,
                self.steps,
                self.seed_token,
                error,
            );
        }
        let planned_footprint = match estimate_sequence_footprint(self) {
            Ok(footprint) => footprint,
            Err(error) => {
                return empty_summary(
                    SmokeStatus::Failed,
                    self.dtype,
                    self.hidden,
                    self.vocab_size,
                    self.steps,
                    self.seed_token,
                    error,
                );
            }
        };
        let memory = crate::smoke::probe::smoke();
        let fits_device_free_memory = memory
            .device_free_memory_bytes
            .map(|free| planned_footprint.device_arena_bytes <= free as u64);
        let uses_declared_descriptors = self
            .weight_plan
            .is_some_and(CudaHfDecodeSequenceWeightPlan::is_declared);
        let ffi_layers = self
            .layers
            .iter()
            .map(|layer| {
                if uses_declared_descriptors {
                    layer.to_descriptor_layout_ffi()
                } else {
                    layer.to_ffi()
                }
            })
            .collect::<Vec<_>>();
        let mut tokens = vec![0u32; self.steps];
        let ffi_request = self.to_ffi(ffi_layers.as_ptr(), tokens.as_mut_ptr());
        let mut out = NervaCudaHfDecodeSequenceResult::default();
        let return_code = run_hf_decode_sequence_u16(&ffi_request, &mut out);
        let status = sequence_status_from_result(return_code, &out);
        tokens.truncate(out.observed_tokens.min(self.steps as u32) as usize);
        let error = (status != SmokeStatus::Ok).then(|| sequence_failure_reason(return_code, &out));
        CudaHfDecodeSequenceSummary {
            status,
            dtype: out.dtype,
            hidden: out.hidden,
            heads: out.heads,
            kv_heads: out.kv_heads,
            head_dim: out.head_dim,
            intermediate: out.intermediate,
            vocab_size: out.vocab_size,
            layer_count: out.layer_count,
            steps: out.steps,
            seed_token: out.seed_token,
            tokens,
            observed_token_hash: out.observed_token_hash,
            planned_footprint,
            device_total_memory_bytes: memory.device_total_memory_bytes,
            device_free_memory_bytes: memory.device_free_memory_bytes,
            fits_device_free_memory,
            resident_weight_bytes: out.resident_weight_bytes,
            planned_weight_blocks: out.planned_weight_blocks,
            planned_gpu_resident_blocks: out.planned_gpu_resident_blocks,
            planned_gpu_staged_blocks: out.planned_gpu_staged_blocks,
            planned_weight_bytes: out.planned_weight_bytes,
            planned_gpu_resident_weight_bytes: out.planned_gpu_resident_weight_bytes,
            planned_gpu_staged_weight_bytes: out.planned_gpu_staged_weight_bytes,
            descriptor_gpu_resident_h2d_bytes: out.descriptor_gpu_resident_h2d_bytes,
            descriptor_gpu_staged_h2d_bytes: out.descriptor_gpu_staged_h2d_bytes,
            planned_weight_descriptor_count: out.planned_weight_descriptor_count,
            planned_weight_descriptor_hash: out.planned_weight_descriptor_hash,
            resident_kv_bytes: out.resident_kv_bytes,
            kv_tokens: out.kv_tokens,
            device_arena_bytes: out.device_arena_bytes,
            pinned_host_bytes: out.pinned_host_bytes,
            h2d_bytes: out.h2d_bytes,
            d2h_bytes: out.d2h_bytes,
            graph_replays: out.graph_replays,
            graph_nodes: out.graph_nodes,
            graph_launches: out.graph_launches,
            graph_captures: out.graph_captures,
            graph_cache_hits: out.graph_cache_hits,
            kernel_launches: out.kernel_launches,
            experimental_rt_selector_launches: out.experimental_rt_selector_launches,
            experimental_rt_sparse_attention_active: out.experimental_rt_sparse_attention_active
                != 0,
            experimental_rt_dense_attention_chunks: out.experimental_rt_dense_attention_chunks,
            experimental_rt_attention_chunks: out.experimental_rt_attention_chunks,
            device_elapsed_ns: out.device_elapsed_ns,
            projection_ns: out.projection_ns,
            qkv_projection_ns: out.qkv_projection_ns,
            attention_output_projection_ns: out.attention_output_projection_ns,
            gate_up_projection_ns: out.gate_up_projection_ns,
            down_projection_ns: out.down_projection_ns,
            lm_head_projection_ns: out.lm_head_projection_ns,
            attention_ns: out.attention_ns,
            mlp_ns: out.mlp_ns,
            norm_ns: out.norm_ns,
            sampling_ns: out.sampling_ns,
            sync_calls: out.sync_calls,
            host_causality_edges: out.host_causality_edges,
            hot_path_allocations: out.hot_path_allocations,
            error,
        }
    }
    fn to_ffi(
        &self,
        layers: *const crate::decode::hf_chain::ffi::NervaCudaHfDecodeChainLayer,
        output_tokens: *mut u32,
    ) -> NervaCudaHfDecodeSequenceRequest {
        let plan = self.weight_plan.unwrap_or_default();
        let descriptors = if self.weight_blocks.is_empty() {
            core::ptr::null()
        } else {
            self.weight_blocks.as_ptr()
        };
        NervaCudaHfDecodeSequenceRequest {
            dtype: self.dtype,
            hidden: self.hidden as u32,
            heads: self.heads as u32,
            kv_heads: self.kv_heads as u32,
            head_dim: self.head_dim as u32,
            intermediate: self.intermediate as u32,
            vocab_size: self.vocab_size as u32,
            layer_count: self.layers.len() as u32,
            steps: self.steps as u32,
            seed_token: self.seed_token,
            prompt_tokens: self.prompt_tokens.as_ptr(),
            prompt_token_count: self.prompt_tokens.len() as u32,
            has_eos_token: self.eos_token.is_some() as u32,
            eos_token: self.eos_token.unwrap_or(0),
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
            planned_weight_descriptors: descriptors,
            planned_weight_descriptor_count: self.weight_blocks.len() as u32,
            planned_weight_descriptor_hash: plan.descriptor_hash,
            output_tokens,
            output_token_capacity: self.steps as u32,
            sampler: self.sampler.to_ffi(),
        }
    }
}
fn planned_ptr(slice: &[u16], plan: CudaHfDecodeSequenceWeightPlan) -> *const u16 {
    if plan.is_declared() {
        core::ptr::null()
    } else {
        slice.as_ptr()
    }
}
