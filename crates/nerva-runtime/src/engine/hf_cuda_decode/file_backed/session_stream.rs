use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_sequence::session::stateful::CudaHfDecodeSequenceLoop;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::causal_lm::types::HfCausalLmStopReason;
use nerva_model::hf::tokenizer::stop_token_ids;

use crate::engine::hf_cuda_decode::file_backed::block_verify::draft_ngram_block;
use crate::engine::hf_cuda_decode::file_backed::progress::HfCudaDeviceSessionChunkProgress;
use crate::engine::hf_cuda_decode::file_backed::projection_mode::HfCudaProjectionMode;
use crate::engine::hf_cuda_decode::file_backed::run::summary_from_sequence;
use crate::engine::hf_cuda_decode::file_backed::session::create_hf_causal_lm_cuda_shard_backed_device_only_session_with_profiling;
use crate::engine::hf_cuda_decode::file_backed::session_stream_queue::BoundedHostOutputQueue;
use crate::engine::hf_cuda_decode::file_backed::session_stream_types::HfCudaDeviceSessionStreamOutput;
use crate::engine::runtime::Runtime;

const BLOCK_VERIFY_INITIAL_PROBE_TOKENS: usize = 2;
const BLOCK_VERIFY_FALLBACK_MIN_CALLS: usize = 1;
const BLOCK_VERIFY_FALLBACK_MIN_DRAFT_TOKENS: usize = 2;
const BLOCK_VERIFY_FALLBACK_ACCEPTED_PER_DRAFT: f64 = 0.60;
const TOKEN_MODE_MAX_ADVANCE_STEPS: usize = 64;
const DECODE_ATTENTION_CHUNK_TOKENS: usize = 64;
const CHUNKED_DECODE_ATTENTION_THRESHOLD: usize = 128;

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
    run_hf_causal_lm_cuda_shard_backed_device_session_stream_with_projection_mode_and_progress(
        runtime,
        dir,
        prompt_tokens,
        max_context_tokens,
        chunk_steps,
        chunks,
        queue_capacity,
        compute_capability,
        HfCudaProjectionMode::Token,
        |_| {},
    )
}

pub fn run_hf_causal_lm_cuda_shard_backed_device_session_stream_with_progress<F>(
    runtime: &Runtime,
    dir: impl AsRef<Path>,
    prompt_tokens: &[TokenId],
    max_context_tokens: usize,
    chunk_steps: usize,
    chunks: usize,
    queue_capacity: usize,
    compute_capability: Option<u32>,
    progress: F,
) -> Result<HfCudaDeviceSessionStreamOutput>
where
    F: FnMut(HfCudaDeviceSessionChunkProgress),
{
    run_hf_causal_lm_cuda_shard_backed_device_session_stream_with_projection_mode_and_progress(
        runtime,
        dir,
        prompt_tokens,
        max_context_tokens,
        chunk_steps,
        chunks,
        queue_capacity,
        compute_capability,
        HfCudaProjectionMode::Token,
        progress,
    )
}

pub fn run_hf_causal_lm_cuda_shard_backed_device_session_stream_with_projection_mode_and_progress<
    F,
>(
    runtime: &Runtime,
    dir: impl AsRef<Path>,
    prompt_tokens: &[TokenId],
    max_context_tokens: usize,
    chunk_steps: usize,
    chunks: usize,
    queue_capacity: usize,
    compute_capability: Option<u32>,
    projection_mode: HfCudaProjectionMode,
    progress: F,
) -> Result<HfCudaDeviceSessionStreamOutput>
where
    F: FnMut(HfCudaDeviceSessionChunkProgress),
{
    run_hf_causal_lm_cuda_shard_backed_device_session_stream_with_projection_mode_profiling_and_progress(
        runtime,
        dir,
        prompt_tokens,
        max_context_tokens,
        chunk_steps,
        chunks,
        queue_capacity,
        compute_capability,
        projection_mode,
        false,
        progress,
    )
}

pub fn run_hf_causal_lm_cuda_shard_backed_device_session_stream_with_projection_mode_profiling_and_progress<
    F,
>(
    runtime: &Runtime,
    dir: impl AsRef<Path>,
    prompt_tokens: &[TokenId],
    max_context_tokens: usize,
    chunk_steps: usize,
    chunks: usize,
    queue_capacity: usize,
    compute_capability: Option<u32>,
    projection_mode: HfCudaProjectionMode,
    detailed_profile: bool,
    mut progress: F,
) -> Result<HfCudaDeviceSessionStreamOutput>
where
    F: FnMut(HfCudaDeviceSessionChunkProgress),
{
    validate_args(prompt_tokens, chunk_steps, chunks, queue_capacity)?;
    let dir = dir.as_ref();
    let load_started = Instant::now();
    let mut session = create_hf_causal_lm_cuda_shard_backed_device_only_session_with_profiling(
        runtime,
        dir,
        max_context_tokens,
        compute_capability,
        detailed_profile,
    )?;
    let load_wall_ns = duration_ns(load_started.elapsed());
    validate_vocab(prompt_tokens, session.metadata.vocab_size)?;
    let stop_tokens = model_stop_tokens(dir, session.metadata.eos_token_id)?;
    let device_stop_token = session
        .metadata
        .eos_token_id
        .or_else(|| stop_tokens.first().copied());
    let prompt = prompt_tokens
        .iter()
        .map(|token| token.0)
        .collect::<Vec<_>>();
    let prefill_started = Instant::now();
    let started = CudaHfDecodeSequenceLoop::start(&mut session.session, &prompt, device_stop_token);
    let prefill_wall_ns = duration_ns(prefill_started.elapsed());
    if started.summary.status != SmokeStatus::Ok {
        return Err(NervaError::InvalidArgument {
            reason: started
                .summary
                .error
                .clone()
                .unwrap_or_else(|| "CUDA HF session stream start failed".to_string()),
        });
    }
    let requested_tokens = requested_token_budget(chunk_steps, chunks, projection_mode);
    progress(HfCudaDeviceSessionChunkProgress::from_prefill_summary(
        requested_tokens,
        prefill_wall_ns,
        &started.summary,
    ));
    let mut loop_state = started.loop_state.unwrap();
    let mut queue = BoundedHostOutputQueue::new(queue_capacity);
    let mut records = Vec::new();
    let mut summaries = Vec::new();
    let mut tokens = Vec::new();
    let mut stop_reason = HfCausalLmStopReason::MaxSteps;
    let mut block_verify_wide_calls = 0usize;
    let mut block_verify_wide_draft_tokens = 0usize;
    let mut block_verify_wide_accepted_tokens = 0usize;
    let mut block_verify_token_fallback = false;
    let mut block_verify_effective_tokens =
        adaptive_block_verify_start_tokens(projection_mode.block_tokens());
    let decode_started = Instant::now();
    for chunk_index in 0..chunks {
        if tokens.len() >= requested_tokens {
            break;
        }
        let remaining_tokens = requested_tokens.saturating_sub(tokens.len());
        let (sequence, current_steps, wide_block_verify) = match projection_mode {
            HfCudaProjectionMode::Token => {
                let current_steps = current_token_mode_steps(
                    prompt_tokens.len(),
                    tokens.len(),
                    remaining_tokens,
                    max_context_tokens,
                    queue_capacity,
                );
                (loop_state.advance(current_steps), current_steps, false)
            }
            HfCudaProjectionMode::BlockVerify { .. } if block_verify_token_fallback => {
                let current_steps = 1.min(remaining_tokens);
                (loop_state.advance(current_steps), current_steps, false)
            }
            HfCudaProjectionMode::BlockVerify { block_tokens } => {
                let current_steps = block_verify_effective_tokens
                    .min(block_tokens)
                    .min(remaining_tokens);
                let draft = draft_ngram_block(
                    prompt_tokens,
                    &tokens,
                    current_steps,
                    session.metadata.vocab_size,
                );
                (
                    loop_state.verify_block(&draft),
                    current_steps,
                    block_tokens > 1 && current_steps > 1,
                )
            }
        };
        let mut summary = summary_from_sequence(&sequence, current_steps)?;
        summary.resident_weights = session.resident_weights.clone();
        if wide_block_verify {
            block_verify_wide_calls = block_verify_wide_calls.saturating_add(1);
            block_verify_wide_draft_tokens =
                block_verify_wide_draft_tokens.saturating_add(current_steps);
            block_verify_wide_accepted_tokens =
                block_verify_wide_accepted_tokens.saturating_add(summary.tokens.len());
            if should_fallback_block_verify(
                block_verify_wide_calls,
                block_verify_wide_draft_tokens,
                block_verify_wide_accepted_tokens,
            ) {
                block_verify_token_fallback = true;
            } else if summary.tokens.len() == current_steps
                && block_verify_effective_tokens < projection_mode.block_tokens()
            {
                block_verify_effective_tokens = block_verify_effective_tokens
                    .saturating_mul(2)
                    .min(projection_mode.block_tokens());
            }
        }
        let hit_stop = contains_stop_token(&summary.tokens, &stop_tokens);
        for (chunk_offset, token) in summary.tokens.iter().copied().enumerate() {
            let record = queue.push(token, chunk_index, chunk_offset)?;
            records.push(record);
        }
        tokens.extend(summary.tokens.iter().copied());
        let observed = summary.tokens.len();
        progress(HfCudaDeviceSessionChunkProgress::from_summary(
            tokens.len(),
            requested_tokens,
            chunk_index,
            hit_stop,
            &summary,
        ));
        summaries.push(summary);
        queue.drain_all();
        if hit_stop {
            stop_reason = HfCausalLmStopReason::EosToken;
            break;
        }
        if observed == 0 {
            break;
        }
        if matches!(projection_mode, HfCudaProjectionMode::Token) && observed < current_steps {
            break;
        }
    }
    let decode_wall_ns = duration_ns(decode_started.elapsed());
    Ok(HfCudaDeviceSessionStreamOutput {
        metadata: session.metadata,
        dtype: session.dtype,
        manifest_entries: session.manifest_entries,
        shard_plan_entries: session.shard_plan_entries,
        tensors_loaded: session.tensors_loaded,
        bytes_loaded: session.bytes_loaded,
        data_hash: session.data_hash,
        data_hash_available: session.data_hash_available,
        projection_mode,
        load_wall_ns,
        prefill_wall_ns,
        decode_wall_ns,
        create: session.create_summary,
        start: started.summary,
        records,
        chunks: summaries,
        tokens,
        queue: queue.summary(),
        stop_reason,
    })
}

fn adaptive_block_verify_start_tokens(block_tokens: usize) -> usize {
    if block_tokens <= 1 {
        1
    } else {
        BLOCK_VERIFY_INITIAL_PROBE_TOKENS.min(block_tokens)
    }
}

fn should_fallback_block_verify(calls: usize, draft_tokens: usize, accepted_tokens: usize) -> bool {
    if calls < BLOCK_VERIFY_FALLBACK_MIN_CALLS
        || draft_tokens < BLOCK_VERIFY_FALLBACK_MIN_DRAFT_TOKENS
    {
        return false;
    }
    if draft_tokens == 0 {
        return false;
    }
    let acceptance = accepted_tokens as f64 / draft_tokens as f64;
    acceptance < BLOCK_VERIFY_FALLBACK_ACCEPTED_PER_DRAFT
}

fn requested_token_budget(
    chunk_steps: usize,
    chunks: usize,
    projection_mode: HfCudaProjectionMode,
) -> usize {
    match projection_mode {
        HfCudaProjectionMode::Token => chunk_steps.saturating_mul(chunks),
        HfCudaProjectionMode::BlockVerify { .. } => chunks,
    }
}

fn current_token_mode_steps(
    prompt_tokens: usize,
    generated_tokens: usize,
    remaining_tokens: usize,
    max_context_tokens: usize,
    queue_capacity: usize,
) -> usize {
    let used_context = prompt_tokens.saturating_add(generated_tokens);
    let context_steps = max_context_tokens
        .saturating_sub(used_context)
        .saturating_add(1);
    let max_steps = TOKEN_MODE_MAX_ADVANCE_STEPS
        .min(remaining_tokens)
        .min(context_steps)
        .min(queue_capacity);
    if max_steps <= 1 {
        return max_steps;
    }
    let kv_tokens = used_context;
    if kv_tokens == 0 {
        return 1;
    }
    let boundary_steps = if kv_tokens <= CHUNKED_DECODE_ATTENTION_THRESHOLD {
        CHUNKED_DECODE_ATTENTION_THRESHOLD
            .saturating_sub(kv_tokens)
            .saturating_add(1)
    } else {
        let chunks = kv_tokens.div_ceil(DECODE_ATTENTION_CHUNK_TOKENS);
        chunks
            .saturating_mul(DECODE_ATTENTION_CHUNK_TOKENS)
            .saturating_sub(kv_tokens)
            .saturating_add(1)
    };
    max_steps.min(boundary_steps.max(1))
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn model_stop_tokens(dir: &Path, metadata_eos: Option<u32>) -> Result<Vec<u32>> {
    let path = dir.to_str().ok_or_else(|| NervaError::InvalidArgument {
        reason: "HF CUDA session stream model path is not valid UTF-8".to_string(),
    })?;
    let mut tokens = stop_token_ids(path).map_err(|err| NervaError::InvalidArgument {
        reason: format!("HF CUDA session stream stop-token discovery failed: {err}"),
    })?;
    if let Some(eos) = metadata_eos {
        tokens.push(eos);
    }
    tokens.sort_unstable();
    tokens.dedup();
    Ok(tokens)
}

fn contains_stop_token(tokens: &[TokenId], stop_tokens: &[u32]) -> bool {
    !stop_tokens.is_empty()
        && tokens
            .iter()
            .any(|token| stop_tokens.binary_search(&token.0).is_ok())
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

#[cfg(test)]
mod tests {
    use super::current_token_mode_steps;

    #[test]
    fn token_mode_batches_until_decode_attention_boundary() {
        assert_eq!(current_token_mode_steps(3965, 0, 512, 8192, 128), 4);
        assert_eq!(current_token_mode_steps(3965, 4, 508, 8192, 128), 64);
        assert_eq!(current_token_mode_steps(3965, 68, 444, 8192, 128), 64);
    }

    #[test]
    fn token_mode_respects_context_queue_and_remaining_budget() {
        assert_eq!(current_token_mode_steps(16, 0, 512, 8192, 128), 64);
        assert_eq!(current_token_mode_steps(120, 0, 512, 8192, 128), 9);
        assert_eq!(current_token_mode_steps(3965, 0, 2, 8192, 128), 2);
        assert_eq!(current_token_mode_steps(3965, 0, 512, 8192, 2), 2);
        assert_eq!(current_token_mode_steps(1, 0, 512, 3, 128), 3);
    }
}
