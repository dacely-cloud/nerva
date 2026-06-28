use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_sequence::session::stateful::CudaHfDecodeSequenceLoop;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::causal_lm::types::HfCausalLmStopReason;
use nerva_model::hf::tokenizer::stop_token_ids;

use crate::engine::hf_cuda_decode::file_backed::progress::HfCudaDeviceSessionChunkProgress;
use crate::engine::hf_cuda_decode::file_backed::run::summary_from_sequence;
use crate::engine::hf_cuda_decode::file_backed::session::create_hf_causal_lm_cuda_shard_backed_device_only_session_with_profiling;
use crate::engine::hf_cuda_decode::file_backed::session_stream_queue::BoundedHostOutputQueue;
use crate::engine::hf_cuda_decode::file_backed::session_stream_types::HfCudaDeviceSessionStreamOutput;
use crate::engine::runtime::Runtime;

const TOKEN_MODE_MAX_ADVANCE_STEPS: usize = 1024;
const TOKEN_MODE_EOS_FIRST_ADVANCE_STEPS: usize = 128;
const TOKEN_MODE_EOS_RAMP_ADVANCE_STEPS: usize = 256;
const TOKEN_MODE_EOS_RAMP_TOKENS: usize = TOKEN_MODE_MAX_ADVANCE_STEPS;

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
    run_hf_causal_lm_cuda_shard_backed_device_session_stream_with_progress(
        runtime,
        dir,
        prompt_tokens,
        max_context_tokens,
        chunk_steps,
        chunks,
        queue_capacity,
        compute_capability,
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
    run_hf_causal_lm_cuda_shard_backed_device_session_stream_with_profiling_and_progress(
        runtime,
        dir,
        prompt_tokens,
        max_context_tokens,
        chunk_steps,
        chunks,
        queue_capacity,
        compute_capability,
        false,
        progress,
    )
}

pub fn run_hf_causal_lm_cuda_shard_backed_device_session_stream_with_profiling_and_progress<F>(
    runtime: &Runtime,
    dir: impl AsRef<Path>,
    prompt_tokens: &[TokenId],
    max_context_tokens: usize,
    chunk_steps: usize,
    chunks: usize,
    queue_capacity: usize,
    compute_capability: Option<u32>,
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
    let requested_tokens = chunk_steps.saturating_mul(chunks);
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
    let decode_started = Instant::now();
    for chunk_index in 0..chunks {
        if tokens.len() >= requested_tokens {
            break;
        }
        let remaining_tokens = requested_tokens.saturating_sub(tokens.len());
        let current_steps = current_token_mode_steps(
            prompt_tokens.len(),
            tokens.len(),
            remaining_tokens,
            max_context_tokens,
            queue_capacity,
            !stop_tokens.is_empty(),
        );
        let sequence = loop_state.advance(current_steps);
        let mut summary = summary_from_sequence(&sequence, current_steps)?;
        summary.resident_weights = session.resident_weights.clone();
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
        if observed < current_steps {
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

fn current_token_mode_steps(
    prompt_tokens: usize,
    generated_tokens: usize,
    remaining_tokens: usize,
    max_context_tokens: usize,
    queue_capacity: usize,
    stop_tokens_known: bool,
) -> usize {
    let used_context = prompt_tokens.saturating_add(generated_tokens);
    let context_steps = max_context_tokens
        .saturating_sub(used_context)
        .saturating_add(1);
    let advance_cap = token_mode_advance_cap(generated_tokens, stop_tokens_known);
    let max_steps = advance_cap
        .min(remaining_tokens)
        .min(context_steps)
        .min(queue_capacity);
    if max_steps <= 1 {
        return max_steps;
    }
    max_steps
}

fn token_mode_advance_cap(generated_tokens: usize, stop_tokens_known: bool) -> usize {
    if !stop_tokens_known || generated_tokens >= TOKEN_MODE_EOS_RAMP_TOKENS {
        return TOKEN_MODE_MAX_ADVANCE_STEPS;
    }
    if generated_tokens == 0 {
        TOKEN_MODE_EOS_FIRST_ADVANCE_STEPS
    } else {
        TOKEN_MODE_EOS_RAMP_ADVANCE_STEPS
    }
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
    fn token_mode_batches_to_runtime_advance_cap() {
        assert_eq!(
            current_token_mode_steps(3965, 0, 2048, 8192, 2048, false),
            1024
        );
        assert_eq!(
            current_token_mode_steps(3965, 1024, 1024, 8192, 2048, false),
            1024
        );
        assert_eq!(
            current_token_mode_steps(3965, 1536, 512, 8192, 2048, false),
            512
        );
    }

    #[test]
    fn token_mode_respects_context_queue_and_remaining_budget() {
        assert_eq!(current_token_mode_steps(16, 0, 512, 8192, 1024, false), 512);
        assert_eq!(
            current_token_mode_steps(120, 0, 2048, 8192, 1024, false),
            1024
        );
        assert_eq!(current_token_mode_steps(3965, 0, 2, 8192, 1024, false), 2);
        assert_eq!(current_token_mode_steps(3965, 0, 512, 8192, 2, false), 2);
        assert_eq!(current_token_mode_steps(1, 0, 512, 3, 1024, false), 3);
    }

    #[test]
    fn token_mode_checks_eos_more_often_before_long_runs() {
        assert_eq!(
            current_token_mode_steps(3965, 0, 2048, 8192, 2048, true),
            128
        );
        assert_eq!(
            current_token_mode_steps(3965, 0, 1024, 8192, 2048, true),
            128
        );
        assert_eq!(
            current_token_mode_steps(3965, 128, 1920, 8192, 2048, true),
            256
        );
        assert_eq!(
            current_token_mode_steps(3965, 896, 1152, 8192, 2048, true),
            256
        );
        assert_eq!(
            current_token_mode_steps(3965, 1152, 896, 8192, 2048, true),
            896
        );
        assert_eq!(
            current_token_mode_steps(3965, 0, 512, 8192, 2048, true),
            128
        );
        assert_eq!(current_token_mode_steps(3965, 0, 64, 8192, 2048, true), 64);
    }
}
