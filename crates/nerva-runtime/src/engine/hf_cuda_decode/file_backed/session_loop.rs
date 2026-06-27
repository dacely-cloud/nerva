use std::path::Path;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_sequence::session::stateful::CudaHfDecodeSequenceLoop;
use nerva_cuda::decode::hf_sequence::session::summary::CudaHfDecodeSequenceSessionCreateSummary;
use nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::hf::metadata::HfModelMetadata;

use crate::engine::hf_cuda_decode::file_backed::run::summary_from_sequence;
use crate::engine::hf_cuda_decode::file_backed::session::{
    HfCudaShardBackedDeviceOnlySession, create_hf_causal_lm_cuda_shard_backed_device_only_session,
};
use crate::engine::hf_cuda_decode::summary::HfCudaSeedDecodeSummary;
use crate::engine::runtime::Runtime;

pub struct HfCudaDeviceSessionLoopOutput {
    pub metadata: HfModelMetadata,
    pub dtype: DType,
    pub manifest_entries: usize,
    pub shard_plan_entries: usize,
    pub tensors_loaded: usize,
    pub bytes_loaded: usize,
    pub data_hash: u64,
    pub data_hash_available: bool,
    pub create: CudaHfDecodeSequenceSessionCreateSummary,
    pub start: CudaHfDecodeSequenceSummary,
    pub chunks: Vec<HfCudaDeviceSessionLoopChunk>,
    pub tokens: Vec<TokenId>,
}

pub struct HfCudaDeviceSessionLoopChunk {
    pub chunk_index: usize,
    pub requested_steps: usize,
    pub summary: HfCudaSeedDecodeSummary,
}

pub fn run_hf_causal_lm_cuda_shard_backed_device_session_loop(
    runtime: &Runtime,
    dir: impl AsRef<Path>,
    prompt_tokens: &[TokenId],
    max_context_tokens: usize,
    chunk_steps: usize,
    chunks: usize,
    compute_capability: Option<u32>,
) -> Result<HfCudaDeviceSessionLoopOutput> {
    validate_loop_args(prompt_tokens, chunk_steps, chunks)?;
    let mut session = create_hf_causal_lm_cuda_shard_backed_device_only_session(
        runtime,
        dir,
        max_context_tokens,
        compute_capability,
    )?;
    run_loop_on_session(&mut session, prompt_tokens, chunk_steps, chunks)
}

fn run_loop_on_session(
    session: &mut HfCudaShardBackedDeviceOnlySession,
    prompt_tokens: &[TokenId],
    chunk_steps: usize,
    chunks: usize,
) -> Result<HfCudaDeviceSessionLoopOutput> {
    validate_vocab(prompt_tokens, session.metadata.vocab_size)?;
    let prompt = prompt_tokens
        .iter()
        .map(|token| token.0)
        .collect::<Vec<_>>();
    let create = session.create_summary.clone();
    let resident_weights = session.resident_weights.clone();
    let started = CudaHfDecodeSequenceLoop::start(
        &mut session.session,
        &prompt,
        session.metadata.eos_token_id,
    );
    if started.summary.status != SmokeStatus::Ok {
        return Err(NervaError::InvalidArgument {
            reason: started
                .summary
                .error
                .clone()
                .unwrap_or_else(|| "CUDA HF stateful session start failed".to_string()),
        });
    }
    let mut loop_state = started.loop_state.unwrap();
    let mut records = Vec::new();
    let mut tokens = Vec::new();
    for chunk_index in 0..chunks {
        let sequence = loop_state.advance(chunk_steps);
        let mut summary = summary_from_sequence(&sequence, chunk_steps)?;
        summary.resident_weights = resident_weights.clone();
        tokens.extend(summary.tokens.iter().copied());
        let observed = summary.tokens.len();
        records.push(HfCudaDeviceSessionLoopChunk {
            chunk_index,
            requested_steps: chunk_steps,
            summary,
        });
        if observed < chunk_steps {
            break;
        }
    }
    Ok(HfCudaDeviceSessionLoopOutput {
        metadata: session.metadata.clone(),
        dtype: session.dtype,
        manifest_entries: session.manifest_entries,
        shard_plan_entries: session.shard_plan_entries,
        tensors_loaded: session.tensors_loaded,
        bytes_loaded: session.bytes_loaded,
        data_hash: session.data_hash,
        data_hash_available: session.data_hash_available,
        create,
        start: started.summary,
        chunks: records,
        tokens,
    })
}

fn validate_loop_args(prompt_tokens: &[TokenId], chunk_steps: usize, chunks: usize) -> Result<()> {
    if prompt_tokens.is_empty() || chunk_steps == 0 || chunks == 0 {
        Err(NervaError::InvalidArgument {
            reason: "HF CUDA session loop requires prompt, chunk steps, and chunks".to_string(),
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
            reason: "HF CUDA session loop prompt token is outside vocabulary".to_string(),
        })
    } else {
        Ok(())
    }
}
