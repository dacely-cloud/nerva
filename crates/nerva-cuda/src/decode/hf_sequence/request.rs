use crate::decode::ffi::CUDA_ERROR_NO_DEVICE;
use crate::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use crate::decode::hf_sequence::ffi::{
    NervaCudaHfDecodeSequenceRequest, NervaCudaHfDecodeSequenceResult, run_hf_decode_sequence_u16,
};
use crate::decode::hf_sequence::summary::{CudaHfDecodeSequenceSummary, empty_summary};
use crate::smoke::status::SmokeStatus;

pub const CUDA_HF_DECODE_SEQUENCE_DTYPE_F16: u32 = 0;
pub const CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16: u32 = 1;

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
    pub eos_token: Option<u32>,
    pub rms_eps: f32,
    pub rope_theta: Option<f32>,
    pub embeddings: &'a [u16],
    pub layers: &'a [CudaHfDecodeChainLayer<'a>],
    pub final_norm_weight: &'a [u16],
    pub lm_head: &'a [u16],
}

impl<'a> CudaHfDecodeSequenceRequest<'a> {
    pub fn run(&self) -> CudaHfDecodeSequenceSummary {
        if let Some(error) = self.validate() {
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
        let ffi_layers = self
            .layers
            .iter()
            .map(|layer| layer.to_ffi())
            .collect::<Vec<_>>();
        let mut tokens = vec![0u32; self.steps];
        let ffi_request = self.to_ffi(ffi_layers.as_ptr(), tokens.as_mut_ptr());
        let mut out = NervaCudaHfDecodeSequenceResult::default();
        let return_code = run_hf_decode_sequence_u16(&ffi_request, &mut out);
        let status = status_from_result(return_code, &out);
        tokens.truncate(out.observed_tokens.min(self.steps as u32) as usize);
        let error = (status != SmokeStatus::Ok).then(|| failure_reason(return_code, &out));
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
            resident_weight_bytes: out.resident_weight_bytes,
            resident_kv_bytes: out.resident_kv_bytes,
            kv_tokens: out.kv_tokens,
            device_arena_bytes: out.device_arena_bytes,
            pinned_host_bytes: out.pinned_host_bytes,
            h2d_bytes: out.h2d_bytes,
            d2h_bytes: out.d2h_bytes,
            graph_replays: out.graph_replays,
            graph_nodes: out.graph_nodes,
            graph_launches: out.graph_launches,
            kernel_launches: out.kernel_launches,
            sync_calls: out.sync_calls,
            host_causality_edges: out.host_causality_edges,
            hot_path_allocations: out.hot_path_allocations,
            error,
        }
    }

    fn validate(&self) -> Option<String> {
        if self.hidden == 0 || self.heads == 0 || self.kv_heads == 0 || self.head_dim == 0 {
            return Some("CUDA HF decode sequence dimensions must be non-zero".to_string());
        }
        if self.vocab_size == 0 || self.intermediate == 0 || self.steps == 0 {
            return Some(
                "CUDA HF decode sequence steps and vocabulary must be non-zero".to_string(),
            );
        }
        if self.layers.is_empty() {
            return Some("CUDA HF decode sequence requires at least one layer".to_string());
        }
        if self.seed_token as usize >= self.vocab_size {
            return Some("CUDA HF decode sequence seed token is outside vocabulary".to_string());
        }
        if self.kv_heads > self.heads || !self.heads.is_multiple_of(self.kv_heads) {
            return Some(
                "CUDA HF decode sequence KV heads must divide attention heads".to_string(),
            );
        }
        if self.dtype > CUDA_HF_DECODE_SEQUENCE_DTYPE_BF16 {
            return Some("CUDA HF decode sequence dtype is unsupported".to_string());
        }
        if self.rope_theta.is_some() && !self.head_dim.is_multiple_of(2) {
            return Some(
                "CUDA HF decode sequence RoPE requires an even head dimension".to_string(),
            );
        }
        self.validate_lengths()
    }

    fn validate_lengths(&self) -> Option<String> {
        if self.embeddings.len() != self.vocab_size * self.hidden {
            return Some(
                "CUDA HF decode sequence embeddings length does not match shape".to_string(),
            );
        }
        if self.final_norm_weight.len() != self.hidden {
            return Some(
                "CUDA HF decode sequence final norm length does not match hidden".to_string(),
            );
        }
        if self.lm_head.len() != self.vocab_size * self.hidden {
            return Some("CUDA HF decode sequence LM head length does not match shape".to_string());
        }
        let attention_hidden = self.heads * self.head_dim;
        let kv_hidden = self.kv_heads * self.head_dim;
        self.layers.iter().enumerate().find_map(|(index, layer)| {
            layer
                .validate(self.hidden, attention_hidden, kv_hidden, self.intermediate)
                .map(|error| format!("layer {index}: {error}"))
        })
    }

    fn to_ffi(
        &self,
        layers: *const crate::decode::hf_chain::ffi::NervaCudaHfDecodeChainLayer,
        output_tokens: *mut u32,
    ) -> NervaCudaHfDecodeSequenceRequest {
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
            has_eos_token: self.eos_token.is_some() as u32,
            eos_token: self.eos_token.unwrap_or(0),
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta.unwrap_or(0.0),
            embeddings: self.embeddings.as_ptr(),
            layers,
            final_norm_weight: self.final_norm_weight.as_ptr(),
            lm_head: self.lm_head.as_ptr(),
            output_tokens,
            output_token_capacity: self.steps as u32,
        }
    }
}

fn status_from_result(return_code: i32, out: &NervaCudaHfDecodeSequenceResult) -> SmokeStatus {
    if return_code == 0 && out.status == 0 {
        SmokeStatus::Ok
    } else if out.cuda_error == CUDA_ERROR_NO_DEVICE || out.device_count == 0 {
        SmokeStatus::Unavailable
    } else {
        SmokeStatus::Failed
    }
}

fn failure_reason(return_code: i32, out: &NervaCudaHfDecodeSequenceResult) -> String {
    format!(
        "CUDA HF decode sequence failed: return_code={return_code} status={} cuda_error={}",
        out.status, out.cuda_error,
    )
}
