use std::path::Path;
use std::time::{Duration, Instant};

use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_sequence::session::request::{
    CudaHfDecodeSequenceBatchAdvanceSummary, CudaHfDecodeSequenceSession,
};
use nerva_cuda::decode::hf_sequence::session::stateful::CudaHfDecodeSequenceLoop;
use nerva_cuda::decode::hf_sequence::session::summary::CudaHfDecodeSequenceSessionCreateSummary;
use nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::hf::metadata::HfModelMetadata;
use nerva_model::hf::tokenizer::stop_token_ids;

use crate::engine::hf_cuda_decode::continuous_batch::{
    CudaDecodeLoopBatchEntry, advance_continuous_decode_batch_once,
};
use crate::engine::hf_cuda_decode::file_backed::session::create_hf_causal_lm_cuda_shard_backed_device_only_session_with_profiling;
use crate::engine::hf_cuda_decode::projection_batch::{
    ProjectionBatchCandidate, ProjectionBatchConfig, ProjectionBatchModelKey,
};
use crate::engine::hf_cuda_decode::summary::HfCudaResidentWeightSummary;
use crate::engine::runtime::Runtime;

pub struct HfCudaSharedForkBatchOutput {
    pub metadata: HfModelMetadata,
    pub dtype: nerva_core::types::dtype::DType,
    pub manifest_entries: usize,
    pub shard_plan_entries: usize,
    pub tensors_loaded: usize,
    pub bytes_loaded: usize,
    pub data_hash: u64,
    pub data_hash_available: bool,
    pub load_wall_ns: u64,
    pub prefill_wall_ns: u64,
    pub first_decode_wall_ns: u64,
    pub continuous_decode_wall_ns: u64,
    pub request_count: usize,
    pub max_new_tokens: usize,
    pub target_block_tokens: usize,
    pub min_block_tokens: usize,
    pub create: CudaHfDecodeSequenceSessionCreateSummary,
    pub fork_creates: Vec<CudaHfDecodeSequenceSessionCreateSummary>,
    pub start_summaries: Vec<CudaHfDecodeSequenceSummary>,
    pub first_token_summaries: Vec<CudaHfDecodeSequenceSummary>,
    pub tokens_by_request: Vec<Vec<TokenId>>,
    pub stopped_by_request: Vec<bool>,
    pub scheduler: HfCudaSharedForkBatchSchedulerSummary,
    pub resident_weights: HfCudaResidentWeightSummary,
}

impl HfCudaSharedForkBatchOutput {
    pub fn total_tokens(&self) -> usize {
        self.tokens_by_request.iter().map(Vec::len).sum()
    }

    pub fn decode_wall_ns(&self) -> u64 {
        self.first_decode_wall_ns
            .saturating_add(self.continuous_decode_wall_ns)
    }

    pub fn tokens_per_second(&self) -> f64 {
        let wall_ns = self.decode_wall_ns();
        if wall_ns == 0 {
            return 0.0;
        }
        self.total_tokens() as f64 * 1_000_000_000.0 / wall_ns as f64
    }

    pub fn used_batched_projection(&self) -> bool {
        self.scheduler.batched_steps > 0
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HfCudaSharedForkBatchSchedulerSummary {
    pub scheduler_steps: usize,
    pub batched_steps: usize,
    pub fallback_steps: usize,
    pub batch_failed_steps: usize,
    pub observed_tokens: usize,
    pub batch_observed_tokens: u64,
    pub fallback_observed_tokens: usize,
    pub batch_projection_elapsed_ns: u64,
    pub batch_qkv_elapsed_ns: u64,
    pub batch_attention_output_elapsed_ns: u64,
    pub batch_gate_up_elapsed_ns: u64,
    pub batch_down_elapsed_ns: u64,
    pub batch_lm_head_elapsed_ns: u64,
    pub batch_pack_kernel_launches: u64,
    pub batch_projection_kernel_launches: u64,
    pub batch_scatter_kernel_launches: u64,
    pub batch_dependency_kernel_launches: u64,
    pub batch_sampling_kernel_launches: u64,
    pub batch_sync_calls: u64,
    pub batch_hot_path_allocations: u64,
    pub last_plan_reason: &'static str,
    pub last_batch_reason: &'static str,
}

pub fn run_hf_causal_lm_cuda_shared_fork_batch_probe(
    runtime: &Runtime,
    dir: impl AsRef<Path>,
    prompt_tokens: &[TokenId],
    request_count: usize,
    max_context_tokens: usize,
    max_new_tokens: usize,
    target_block_tokens: usize,
    min_block_tokens: usize,
    compute_capability: Option<u32>,
    detailed_profile: bool,
) -> Result<HfCudaSharedForkBatchOutput> {
    validate_args(
        prompt_tokens,
        request_count,
        max_context_tokens,
        max_new_tokens,
    )?;
    let dir = dir.as_ref();
    let load_started = Instant::now();
    let mut root = create_hf_causal_lm_cuda_shard_backed_device_only_session_with_profiling(
        runtime,
        dir,
        max_context_tokens,
        compute_capability,
        detailed_profile,
    )?;
    let load_wall_ns = duration_ns(load_started.elapsed());
    validate_vocab(prompt_tokens, root.metadata.vocab_size)?;

    let metadata = root.metadata.clone();
    let dtype = root.dtype;
    let manifest_entries = root.manifest_entries;
    let shard_plan_entries = root.shard_plan_entries;
    let tensors_loaded = root.tensors_loaded;
    let bytes_loaded = root.bytes_loaded;
    let data_hash = root.data_hash;
    let data_hash_available = root.data_hash_available;
    let create = root.create_summary.clone();
    let resident_weights = root.resident_weights.clone();
    let model = shared_fork_projection_model_key(
        &metadata,
        dtype,
        data_hash,
        data_hash_available,
        resident_weights.plan_descriptor_hash,
    );
    let stop_tokens = model_stop_tokens(dir, metadata.eos_token_id)?;
    let device_stop_token = metadata
        .eos_token_id
        .or_else(|| stop_tokens.first().copied());

    let mut sessions = Vec::with_capacity(request_count);
    let mut fork_creates = Vec::with_capacity(request_count.saturating_sub(1));
    for _ in 1..request_count {
        let forked = root.session.fork_shared_weights(detailed_profile);
        if forked.summary.status != SmokeStatus::Ok {
            return Err(NervaError::InvalidArgument {
                reason: forked
                    .summary
                    .error
                    .clone()
                    .unwrap_or_else(|| "CUDA HF shared-weight fork failed".to_string()),
            });
        }
        fork_creates.push(forked.summary);
        sessions.push(forked.session.unwrap());
    }
    sessions.insert(0, root.session);

    let prompt = prompt_tokens
        .iter()
        .map(|token| token.0)
        .collect::<Vec<_>>();
    let mut tokens_by_request = vec![Vec::new(); request_count];
    let mut stopped_by_request = vec![false; request_count];
    let prefill_started = Instant::now();
    let started = start_loops(&mut sessions, &prompt, device_stop_token)?;
    let prefill_wall_ns = duration_ns(prefill_started.elapsed());
    let mut loops = started.loops;

    let first_decode_started = Instant::now();
    let first_token_summaries = drain_first_tokens(
        &mut loops,
        &stop_tokens,
        max_new_tokens,
        &mut tokens_by_request,
        &mut stopped_by_request,
    )?;
    let first_decode_wall_ns = duration_ns(first_decode_started.elapsed());

    let continuous_decode_started = Instant::now();
    let scheduler = advance_continuous_until_done(
        &mut loops,
        &model,
        prompt_tokens.len(),
        max_new_tokens,
        max_context_tokens,
        target_block_tokens,
        min_block_tokens,
        &stop_tokens,
        &mut tokens_by_request,
        &mut stopped_by_request,
    );
    let continuous_decode_wall_ns = duration_ns(continuous_decode_started.elapsed());

    Ok(HfCudaSharedForkBatchOutput {
        metadata,
        dtype,
        manifest_entries,
        shard_plan_entries,
        tensors_loaded,
        bytes_loaded,
        data_hash,
        data_hash_available,
        load_wall_ns,
        prefill_wall_ns,
        first_decode_wall_ns,
        continuous_decode_wall_ns,
        request_count,
        max_new_tokens,
        target_block_tokens: target_block_tokens.max(1),
        min_block_tokens: min_block_tokens.max(1),
        create,
        fork_creates,
        start_summaries: started.summaries,
        first_token_summaries,
        tokens_by_request,
        stopped_by_request,
        scheduler,
        resident_weights,
    })
}

struct StartedLoops<'a> {
    loops: Vec<CudaHfDecodeSequenceLoop<'a>>,
    summaries: Vec<CudaHfDecodeSequenceSummary>,
}

fn start_loops<'a>(
    sessions: &'a mut [CudaHfDecodeSequenceSession],
    prompt_tokens: &[u32],
    device_stop_token: Option<u32>,
) -> Result<StartedLoops<'a>> {
    let mut loops = Vec::with_capacity(sessions.len());
    let mut summaries = Vec::with_capacity(sessions.len());
    for session in sessions {
        let started = CudaHfDecodeSequenceLoop::start(session, prompt_tokens, device_stop_token);
        if started.summary.status != SmokeStatus::Ok {
            return Err(NervaError::InvalidArgument {
                reason: started
                    .summary
                    .error
                    .clone()
                    .unwrap_or_else(|| "CUDA HF shared fork batch prefill failed".to_string()),
            });
        }
        summaries.push(started.summary);
        loops.push(started.loop_state.unwrap());
    }
    Ok(StartedLoops { loops, summaries })
}

fn drain_first_tokens(
    loops: &mut [CudaHfDecodeSequenceLoop<'_>],
    stop_tokens: &[u32],
    max_new_tokens: usize,
    tokens_by_request: &mut [Vec<TokenId>],
    stopped_by_request: &mut [bool],
) -> Result<Vec<CudaHfDecodeSequenceSummary>> {
    let mut summaries = Vec::with_capacity(loops.len());
    for (request_index, loop_state) in loops.iter_mut().enumerate() {
        let summary = loop_state.advance(1);
        if summary.status != SmokeStatus::Ok {
            return Err(NervaError::InvalidArgument {
                reason: summary
                    .error
                    .clone()
                    .unwrap_or_else(|| "CUDA HF shared fork batch first token failed".to_string()),
            });
        }
        for token in summary.tokens.iter().copied() {
            tokens_by_request[request_index].push(TokenId(token));
        }
        stopped_by_request[request_index] = tokens_by_request[request_index].len()
            >= max_new_tokens
            || summary.tokens.is_empty()
            || contains_stop_token(&tokens_by_request[request_index], stop_tokens);
        summaries.push(summary);
    }
    Ok(summaries)
}

fn advance_continuous_until_done(
    loops: &mut [CudaHfDecodeSequenceLoop<'_>],
    model: &ProjectionBatchModelKey,
    prompt_tokens: usize,
    max_new_tokens: usize,
    max_context_tokens: usize,
    target_block_tokens: usize,
    min_block_tokens: usize,
    stop_tokens: &[u32],
    tokens_by_request: &mut [Vec<TokenId>],
    stopped_by_request: &mut [bool],
) -> HfCudaSharedForkBatchSchedulerSummary {
    let config = ProjectionBatchConfig::new(target_block_tokens, min_block_tokens);
    let mut summary = HfCudaSharedForkBatchSchedulerSummary::default();
    while has_live_requests(tokens_by_request, stopped_by_request, max_new_tokens) {
        let entries = loops
            .iter_mut()
            .enumerate()
            .filter_map(|(index, loop_state)| {
                if stopped_by_request[index] || tokens_by_request[index].len() >= max_new_tokens {
                    return None;
                }
                Some(CudaDecodeLoopBatchEntry {
                    candidate: ProjectionBatchCandidate {
                        request_id: index as u64,
                        model: *model,
                        prompt_tokens,
                        generated_tokens: tokens_by_request[index].len(),
                        remaining_tokens: max_new_tokens
                            .saturating_sub(tokens_by_request[index].len()),
                        max_context_tokens,
                        ready: true,
                        stopped: false,
                    },
                    loop_state,
                })
            })
            .collect::<Vec<_>>();
        if entries.is_empty() {
            break;
        }

        let output = advance_continuous_decode_batch_once(entries, config);
        summary.scheduler_steps += 1;
        summary.last_plan_reason = output.plan.projection.reason.as_str();
        if output.used_batched_projection() {
            summary.batched_steps += 1;
        }
        if output.fallback.is_some() {
            summary.fallback_steps += 1;
        }
        if let Some(selected) = output.selected.as_ref() {
            match selected.mode {
                crate::engine::hf_cuda_decode::batch_advance::CudaDecodeBatchAdvanceMode::BatchFailed { reason } => {
                    summary.batch_failed_steps += 1;
                    summary.last_batch_reason = reason;
                }
                crate::engine::hf_cuda_decode::batch_advance::CudaDecodeBatchAdvanceMode::FallbackSequential { reason } => {
                    summary.last_batch_reason = reason;
                }
                crate::engine::hf_cuda_decode::batch_advance::CudaDecodeBatchAdvanceMode::Batched => {
                    summary.last_batch_reason = "batched";
                }
            }
            if let Some(batch) = selected.batch.as_ref() {
                accumulate_batch_summary(&mut summary, batch);
            }
        }
        if let Some(fallback) = output.fallback.as_ref() {
            summary.fallback_observed_tokens += fallback.observed_tokens();
        }

        let mut observed_this_step = 0usize;
        for record in output.records {
            let request_index = record.request_id as usize;
            if request_index >= tokens_by_request.len() {
                continue;
            }
            observed_this_step += record.tokens.len();
            let no_tokens = record.tokens.is_empty();
            for token in record.tokens {
                if tokens_by_request[request_index].len() < max_new_tokens {
                    tokens_by_request[request_index].push(TokenId(token));
                }
            }
            stopped_by_request[request_index] = tokens_by_request[request_index].len()
                >= max_new_tokens
                || no_tokens
                || contains_stop_token(&tokens_by_request[request_index], stop_tokens);
        }
        summary.observed_tokens += observed_this_step;
        if observed_this_step == 0 {
            break;
        }
    }
    summary
}

fn accumulate_batch_summary(
    out: &mut HfCudaSharedForkBatchSchedulerSummary,
    batch: &CudaHfDecodeSequenceBatchAdvanceSummary,
) {
    out.batch_observed_tokens += batch.observed_tokens as u64;
    out.batch_projection_elapsed_ns += batch.projection_elapsed_ns;
    out.batch_qkv_elapsed_ns += batch.qkv_elapsed_ns;
    out.batch_attention_output_elapsed_ns += batch.attention_output_elapsed_ns;
    out.batch_gate_up_elapsed_ns += batch.gate_up_elapsed_ns;
    out.batch_down_elapsed_ns += batch.down_elapsed_ns;
    out.batch_lm_head_elapsed_ns += batch.lm_head_elapsed_ns;
    out.batch_pack_kernel_launches += batch.pack_kernel_launches;
    out.batch_projection_kernel_launches += batch.projection_kernel_launches;
    out.batch_scatter_kernel_launches += batch.scatter_kernel_launches;
    out.batch_dependency_kernel_launches += batch.dependency_kernel_launches;
    out.batch_sampling_kernel_launches += batch.sampling_kernel_launches;
    out.batch_sync_calls += batch.sync_calls;
    out.batch_hot_path_allocations += batch.hot_path_allocations;
}

fn has_live_requests(
    tokens_by_request: &[Vec<TokenId>],
    stopped_by_request: &[bool],
    max_new_tokens: usize,
) -> bool {
    tokens_by_request
        .iter()
        .zip(stopped_by_request.iter().copied())
        .any(|(tokens, stopped)| !stopped && tokens.len() < max_new_tokens)
}

fn shared_fork_projection_model_key(
    metadata: &HfModelMetadata,
    dtype: nerva_core::types::dtype::DType,
    data_hash: u64,
    data_hash_available: bool,
    descriptor_hash: u64,
) -> ProjectionBatchModelKey {
    // Direct shared-weight forks prove common resident weights even when the
    // checkpoint loader cannot provide a content hash for the source shards.
    let shared_weight_hash = if data_hash_available {
        data_hash
    } else {
        descriptor_hash
    };
    ProjectionBatchModelKey {
        data_hash: shared_weight_hash,
        data_hash_available: true,
        dtype,
        hidden_size: metadata.hidden_size,
        attention_heads: metadata.num_attention_heads,
        kv_heads: metadata.num_key_value_heads,
        head_dim: metadata.head_dim(),
        intermediate_size: metadata.intermediate_size,
        vocab_size: metadata.vocab_size,
        layer_count: metadata.num_hidden_layers,
    }
}

fn model_stop_tokens(dir: &Path, metadata_eos: Option<u32>) -> Result<Vec<u32>> {
    let path = dir.to_str().ok_or_else(|| NervaError::InvalidArgument {
        reason: "HF CUDA shared fork batch model path is not valid UTF-8".to_string(),
    })?;
    let mut tokens = stop_token_ids(path).map_err(|err| NervaError::InvalidArgument {
        reason: format!("HF CUDA shared fork batch stop-token discovery failed: {err}"),
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
    request_count: usize,
    max_context_tokens: usize,
    max_new_tokens: usize,
) -> Result<()> {
    if prompt_tokens.is_empty()
        || request_count == 0
        || max_context_tokens == 0
        || max_new_tokens == 0
    {
        Err(NervaError::InvalidArgument {
            reason: "HF CUDA shared fork batch requires prompt, requests, context, and new tokens"
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
            reason: "HF CUDA shared fork batch prompt token is outside vocabulary".to_string(),
        })
    } else {
        Ok(())
    }
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}
