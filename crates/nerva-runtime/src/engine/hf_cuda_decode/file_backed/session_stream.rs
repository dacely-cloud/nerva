use std::path::Path;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_sequence::session::stateful::CudaHfDecodeSequenceLoop;
use nerva_cuda::decode::hf_sequence::session::summary::CudaHfDecodeSequenceSessionCreateSummary;
use nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::causal_lm::types::HfCausalLmStopReason;
use nerva_model::hf::metadata::HfModelMetadata;

use crate::engine::hf_cuda_decode::file_backed::run::summary_from_sequence;
use crate::engine::hf_cuda_decode::file_backed::session::create_hf_causal_lm_cuda_shard_backed_device_only_session;
use crate::engine::hf_cuda_decode::file_backed::session_stream_queue::BoundedHostOutputQueue;
use crate::engine::hf_cuda_decode::summary::HfCudaSeedDecodeSummary;
use crate::engine::runtime::Runtime;

pub struct HfCudaDeviceSessionStreamOutput {
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
    pub records: Vec<HfCudaDeviceSessionStreamRecord>,
    pub chunks: Vec<HfCudaSeedDecodeSummary>,
    pub tokens: Vec<TokenId>,
    pub queue: HfCudaHostOutputQueueSummary,
    pub stop_reason: HfCausalLmStopReason,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HfCudaDeviceSessionStreamRecord {
    pub token_index: u64,
    pub token: TokenId,
    pub chunk_index: usize,
    pub chunk_offset: usize,
    pub queue_slot: usize,
    pub host_visible_order: u64,
    pub device_authoritative: bool,
    pub host_causality_edge: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HfCudaHostOutputQueueSummary {
    pub capacity: usize,
    pub pushes: u64,
    pub drains: u64,
    pub high_watermark: usize,
    pub overflow_rejections: u64,
    pub host_causality_edges: u64,
}

pub fn run_hf_causal_lm_cuda_shard_backed_device_session_stream(
    runtime: &Runtime,
    dir: impl AsRef<Path>,
    prompt_tokens: &[TokenId],
    max_context_tokens: usize,
    chunk_steps: usize,
    chunks: usize,
    queue_capacity: usize,
    compute_capability: Option<u32>,
) -> Result<HfCudaDeviceSessionStreamOutput> {
    validate_args(prompt_tokens, chunk_steps, chunks, queue_capacity)?;
    let mut session = create_hf_causal_lm_cuda_shard_backed_device_only_session(
        runtime,
        dir,
        max_context_tokens,
        compute_capability,
    )?;
    validate_vocab(prompt_tokens, session.metadata.vocab_size)?;
    let prompt = prompt_tokens
        .iter()
        .map(|token| token.0)
        .collect::<Vec<_>>();
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
                .unwrap_or_else(|| "CUDA HF session stream start failed".to_string()),
        });
    }
    let mut loop_state = started.loop_state.unwrap();
    let mut queue = BoundedHostOutputQueue::new(queue_capacity);
    let mut records = Vec::new();
    let mut summaries = Vec::new();
    let mut tokens = Vec::new();
    let mut stop_reason = HfCausalLmStopReason::MaxSteps;
    for chunk_index in 0..chunks {
        let sequence = loop_state.advance(chunk_steps);
        let mut summary = summary_from_sequence(&sequence, chunk_steps)?;
        summary.resident_weights = session.resident_weights.clone();
        let hit_eos = contains_eos(&summary.tokens, session.metadata.eos_token_id);
        for (chunk_offset, token) in summary.tokens.iter().copied().enumerate() {
            let record = queue.push(token, chunk_index, chunk_offset)?;
            records.push(record);
        }
        tokens.extend(summary.tokens.iter().copied());
        let observed = summary.tokens.len();
        summaries.push(summary);
        queue.drain_all();
        if hit_eos {
            stop_reason = HfCausalLmStopReason::EosToken;
            break;
        }
        if observed < chunk_steps {
            break;
        }
    }
    Ok(HfCudaDeviceSessionStreamOutput {
        metadata: session.metadata,
        dtype: session.dtype,
        manifest_entries: session.manifest_entries,
        shard_plan_entries: session.shard_plan_entries,
        tensors_loaded: session.tensors_loaded,
        bytes_loaded: session.bytes_loaded,
        data_hash: session.data_hash,
        data_hash_available: session.data_hash_available,
        create: session.create_summary,
        start: started.summary,
        records,
        chunks: summaries,
        tokens,
        queue: queue.summary(),
        stop_reason,
    })
}

fn contains_eos(tokens: &[TokenId], eos_token: Option<u32>) -> bool {
    eos_token.is_some_and(|eos| tokens.iter().any(|token| token.0 == eos))
}

fn validate_args(
    prompt_tokens: &[TokenId],
    chunk_steps: usize,
    chunks: usize,
    queue_capacity: usize,
) -> Result<()> {
    if prompt_tokens.is_empty() || chunk_steps == 0 || chunks == 0 || queue_capacity == 0 {
        Err(NervaError::InvalidArgument {
            reason: "HF CUDA session stream requires prompt, chunks, and queue capacity"
                .to_string(),
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
            reason: "HF CUDA session stream prompt token is outside vocabulary".to_string(),
        })
    } else {
        Ok(())
    }
}
