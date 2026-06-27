use std::path::Path;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_sequence::session::request::{
    CudaHfDecodeSequenceSession, CudaHfDecodeSequenceSessionConfig,
};
use nerva_cuda::decode::hf_sequence::session::summary::CudaHfDecodeSequenceSessionCreateSummary;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::hf::metadata::HfModelMetadata;

use crate::engine::hf_cuda_decode::contract::cuda_weight_plan;
use crate::engine::hf_cuda_decode::file_backed::descriptors::{
    descriptor_marker_layers, shard_backed_resident_weights,
};
use crate::engine::hf_cuda_decode::file_backed::load::load_shard_backed_weights;
use crate::engine::hf_cuda_decode::file_backed::run::summary_from_sequence;
use crate::engine::hf_cuda_decode::sequence::cuda_dtype;
use crate::engine::hf_cuda_decode::summary::{
    HfCudaResidentWeightSummary, HfCudaSeedDecodeSummary,
};
use crate::engine::runtime::Runtime;

pub struct HfCudaShardBackedDeviceOnlySession {
    pub metadata: HfModelMetadata,
    pub dtype: DType,
    pub manifest_entries: usize,
    pub shard_plan_entries: usize,
    pub tensors_loaded: usize,
    pub bytes_loaded: usize,
    pub data_hash: u64,
    pub data_hash_available: bool,
    pub create_summary: CudaHfDecodeSequenceSessionCreateSummary,
    pub(in crate::engine::hf_cuda_decode::file_backed) resident_weights:
        HfCudaResidentWeightSummary,
    pub(in crate::engine::hf_cuda_decode::file_backed) session: CudaHfDecodeSequenceSession,
}

impl HfCudaShardBackedDeviceOnlySession {
    pub fn run(
        &mut self,
        prompt_tokens: &[TokenId],
        steps: usize,
    ) -> Result<HfCudaSeedDecodeSummary> {
        validate_prompt(prompt_tokens, steps)?;
        validate_vocab(prompt_tokens, self.metadata.vocab_size)?;
        let prompt = prompt_tokens
            .iter()
            .map(|token| token.0)
            .collect::<Vec<_>>();
        let sequence = self.session.run(&prompt, steps, self.metadata.eos_token_id);
        let mut summary = summary_from_sequence(&sequence, steps)?;
        summary.resident_weights = self.resident_weights.clone();
        Ok(summary)
    }
}

pub fn create_hf_causal_lm_cuda_shard_backed_device_only_session(
    runtime: &Runtime,
    dir: impl AsRef<Path>,
    max_context_tokens: usize,
    compute_capability: Option<u32>,
) -> Result<HfCudaShardBackedDeviceOnlySession> {
    if max_context_tokens == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "HF CUDA shard-backed session capacity must be non-zero".to_string(),
        });
    }
    let compute_capability = compute_capability.or_else(discovered_cuda_compute_capability);
    let weights = load_shard_backed_weights(dir.as_ref())?;
    let resident_weights = shard_backed_resident_weights(runtime, &weights, compute_capability)?;
    let weight_plan = cuda_weight_plan(&resident_weights.summary, &resident_weights.descriptors)?;
    let layers = descriptor_marker_layers(&weights.metadata);
    let created = CudaHfDecodeSequenceSessionConfig {
        dtype: cuda_dtype(weights.dtype)?,
        hidden: weights.metadata.hidden_size,
        heads: weights.metadata.num_attention_heads,
        kv_heads: weights.metadata.num_key_value_heads,
        head_dim: weights.metadata.head_dim(),
        intermediate: weights.metadata.intermediate_size,
        vocab_size: weights.metadata.vocab_size,
        max_context_tokens,
        rms_eps: weights.metadata.rms_norm_eps.unwrap_or(1e-5),
        rope_theta: weights.metadata.rope_theta,
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(weight_plan),
        weight_blocks: &resident_weights.descriptors,
    }
    .create();
    if created.summary.status != SmokeStatus::Ok {
        return Err(NervaError::InvalidArgument {
            reason: created
                .summary
                .error
                .clone()
                .unwrap_or_else(|| "CUDA HF shard-backed session create failed".to_string()),
        });
    }
    let mut resident_summary = resident_weights.summary;
    attach_create_contract(&mut resident_summary, &created.summary);
    Ok(HfCudaShardBackedDeviceOnlySession {
        metadata: weights.metadata,
        dtype: weights.dtype,
        manifest_entries: weights.manifest.entries.len(),
        shard_plan_entries: weights.shard_plan.entries.len(),
        tensors_loaded: weights.manifest.entries.len(),
        bytes_loaded: weights.manifest.total_weight_bytes,
        data_hash: weights.data_hash,
        data_hash_available: weights.data_hash_available,
        create_summary: created.summary,
        resident_weights: resident_summary,
        session: created.session.unwrap(),
    })
}

fn attach_create_contract(
    summary: &mut HfCudaResidentWeightSummary,
    create: &CudaHfDecodeSequenceSessionCreateSummary,
) {
    summary.cuda_contract_blocks = create.planned_weight_blocks as u64;
    summary.cuda_contract_weight_bytes = create.planned_weight_bytes;
    summary.cuda_contract_descriptor_blocks = create.planned_weight_descriptor_count as u64;
    summary.cuda_contract_descriptor_hash = create.planned_weight_descriptor_hash;
    summary.cuda_contract_gpu_resident_h2d_bytes = create.descriptor_gpu_resident_h2d_bytes;
    summary.cuda_contract_gpu_staged_h2d_bytes = create.descriptor_gpu_staged_h2d_bytes;
    summary.cuda_contract_matched = summary.plan_descriptor_blocks
        == summary.cuda_contract_descriptor_blocks
        && summary.plan_descriptor_hash == summary.cuda_contract_descriptor_hash
        && summary.plan_weight_bytes == summary.cuda_contract_weight_bytes;
}

fn validate_prompt(prompt_tokens: &[TokenId], steps: usize) -> Result<()> {
    if steps == 0 || prompt_tokens.is_empty() {
        Err(NervaError::InvalidArgument {
            reason: "HF CUDA shard-backed session run requires prompt and steps".to_string(),
        })
    } else {
        Ok(())
    }
}

fn validate_vocab(prompt_tokens: &[TokenId], vocab_size: usize) -> Result<()> {
    if prompt_tokens
        .iter()
        .any(|token| token.0 as usize >= vocab_size)
    {
        Err(NervaError::InvalidArgument {
            reason: "HF CUDA shard-backed session prompt token is outside vocabulary".to_string(),
        })
    } else {
        Ok(())
    }
}

fn discovered_cuda_compute_capability() -> Option<u32> {
    let summary = crate::capabilities::discovery::cuda_smoke();
    if summary.status != SmokeStatus::Ok {
        return None;
    }
    let major = u32::try_from(summary.compute_capability_major?).ok()?;
    let minor = u32::try_from(summary.compute_capability_minor?).ok()?;
    Some(major * 10 + minor)
}
