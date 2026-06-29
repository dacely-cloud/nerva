use std::path::Path;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_sequence::request::{
    CudaHfDecodeSamplerConfig, CudaHfDecodeSequenceRequest,
};
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::hf::metadata::HfModelMetadata;

use crate::engine::hf_cuda_decode::contract::{attach_cuda_weight_contract, cuda_weight_plan};
use crate::engine::hf_cuda_decode::file_backed::descriptors::{
    descriptor_marker_layers, shard_backed_resident_weights,
};
use crate::engine::hf_cuda_decode::file_backed::load::load_shard_backed_weights;
use crate::engine::hf_cuda_decode::sequence::cuda_dtype;
use crate::engine::hf_cuda_decode::sequence_ledger::sequence_ledgers;
use crate::engine::hf_cuda_decode::summary::HfCudaSeedDecodeSummary;
use crate::engine::hf_cuda_decode::totals::{CudaDecodeCounters, DecodeParts, build_summary};
use crate::engine::runtime::Runtime;

pub struct HfCudaShardBackedDeviceOnlyOutput {
    pub metadata: HfModelMetadata,
    pub dtype: DType,
    pub manifest_entries: usize,
    pub shard_plan_entries: usize,
    pub tensors_loaded: usize,
    pub bytes_loaded: usize,
    pub data_hash: u64,
    pub data_hash_available: bool,
    pub summary: HfCudaSeedDecodeSummary,
}

pub fn run_hf_causal_lm_cuda_shard_backed_device_only(
    runtime: &Runtime,
    dir: impl AsRef<Path>,
    prompt_tokens: &[TokenId],
    steps: usize,
    compute_capability: Option<u32>,
) -> Result<HfCudaShardBackedDeviceOnlyOutput> {
    validate_prompt(prompt_tokens, steps)?;
    let compute_capability = compute_capability.or_else(discovered_cuda_compute_capability);
    if compute_capability.is_none() {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "HF CUDA shard-backed device decode could not discover CUDA compute capability: {}",
                crate::capabilities::discovery::cuda_smoke().to_json()
            ),
        });
    }
    let weights = load_shard_backed_weights(dir.as_ref())?;
    validate_vocab(prompt_tokens, weights.metadata.vocab_size)?;
    let resident_weights = shard_backed_resident_weights(runtime, &weights, compute_capability)?;
    let weight_plan = cuda_weight_plan(&resident_weights.summary, &resident_weights.descriptors)?;
    let prompt_token_ids = prompt_tokens
        .iter()
        .map(|token| token.0)
        .collect::<Vec<_>>();
    let layers = descriptor_marker_layers(&weights.metadata);
    let sequence = CudaHfDecodeSequenceRequest {
        dtype: cuda_dtype(weights.dtype)?,
        hidden: weights.metadata.hidden_size,
        heads: weights.metadata.num_attention_heads,
        kv_heads: weights.metadata.num_key_value_heads,
        head_dim: weights.metadata.head_dim(),
        intermediate: weights.metadata.intermediate_size,
        vocab_size: weights.metadata.vocab_size,
        steps,
        seed_token: prompt_tokens.last().unwrap().0,
        prompt_tokens: &prompt_token_ids,
        eos_token: weights.metadata.eos_token_id,
        rms_eps: weights.metadata.rms_norm_eps.unwrap_or(1e-5),
        rope_theta: weights.metadata.rope_theta,
        embeddings: &[],
        layers: &layers,
        final_norm_weight: &[],
        lm_head: &[],
        weight_plan: Some(weight_plan),
        weight_blocks: &resident_weights.descriptors,
        sampler: CudaHfDecodeSamplerConfig::greedy(),
    }
    .run();
    let mut summary = summary_from_sequence(&sequence, steps)?;
    let mut resident_summary = resident_weights.summary;
    attach_cuda_weight_contract(&mut resident_summary, &sequence)?;
    summary.hot_path_allocations += resident_summary.hot_path_allocations;
    summary.resident_weights = resident_summary;
    Ok(HfCudaShardBackedDeviceOnlyOutput {
        metadata: weights.metadata,
        dtype: weights.dtype,
        manifest_entries: weights.manifest.entries.len(),
        shard_plan_entries: weights.shard_plan.entries.len(),
        tensors_loaded: weights.manifest.entries.len(),
        bytes_loaded: weights.manifest.total_weight_bytes,
        data_hash: weights.data_hash,
        data_hash_available: weights.data_hash_available,
        summary,
    })
}

pub(super) fn summary_from_sequence(
    sequence: &nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary,
    steps: usize,
) -> Result<HfCudaSeedDecodeSummary> {
    let mut counters = CudaDecodeCounters::default();
    counters.record_sequence(sequence);
    if sequence.status != SmokeStatus::Ok {
        return Err(NervaError::InvalidArgument {
            reason: sequence
                .error
                .clone()
                .unwrap_or_else(|| "CUDA HF shard-backed device sequence failed".to_string()),
        });
    }
    let tokens = sequence
        .tokens
        .iter()
        .copied()
        .map(TokenId)
        .collect::<Vec<_>>();
    let ledgers = sequence_ledgers(sequence);
    for ledger in &ledgers {
        ledger.require_zero_hot_path_allocations()?;
    }
    Ok(build_summary(
        SmokeStatus::Ok,
        DecodeParts::new(
            steps,
            tokens,
            Vec::new(),
            "device_only_unverified",
            false,
            ledgers,
        ),
        &[],
        counters,
        None,
    ))
}

fn validate_prompt(prompt_tokens: &[TokenId], steps: usize) -> Result<()> {
    if steps == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "HF CUDA shard-backed device decode steps must be non-zero".to_string(),
        });
    }
    if prompt_tokens.is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: "HF CUDA shard-backed device decode requires prompt tokens".to_string(),
        });
    }
    Ok(())
}

fn validate_vocab(prompt_tokens: &[TokenId], vocab_size: usize) -> Result<()> {
    if prompt_tokens
        .iter()
        .any(|token| token.0 as usize >= vocab_size)
    {
        Err(NervaError::InvalidArgument {
            reason: "HF CUDA shard-backed prompt token is outside vocabulary".to_string(),
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
