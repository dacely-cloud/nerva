use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::causal_lm::types::{HfCausalLmDecodeScratch, HfCausalLmLoaded, HfCausalLmModel};

use crate::engine::hf_cuda_decode::resident::loaded_resident_weight_summary;
use crate::engine::hf_cuda_decode::sequence::run_device_sequence;
use crate::engine::hf_cuda_decode::sequence_ledger::sequence_ledgers;
use crate::engine::hf_cuda_decode::summary::HfCudaSeedDecodeSummary;
use crate::engine::hf_cuda_decode::totals::{CudaDecodeCounters, DecodeParts, build_summary};
use crate::engine::runtime::Runtime;

pub fn run_hf_causal_lm_cuda_seed_decode(
    model: &HfCausalLmModel,
    seed: TokenId,
    steps: usize,
) -> Result<HfCudaSeedDecodeSummary> {
    run_hf_causal_lm_cuda_prompt_decode(model, &[seed], steps)
}

pub fn run_hf_causal_lm_cuda_prompt_decode(
    model: &HfCausalLmModel,
    prompt_tokens: &[TokenId],
    steps: usize,
) -> Result<HfCudaSeedDecodeSummary> {
    if steps == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "HF CUDA seed decode steps must be non-zero".to_string(),
        });
    }
    if prompt_tokens.is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: "HF CUDA prompt decode requires prompt tokens".to_string(),
        });
    }
    let context_tokens =
        steps
            .checked_add(prompt_tokens.len())
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: "HF CUDA seed decode context length overflow".to_string(),
            })?;
    let mut cpu_scratch = HfCausalLmDecodeScratch::new_with_context(
        model.shape(),
        model.metadata().vocab_size,
        model.layer_count(),
        context_tokens,
    )?;
    let output = model.decode_greedy_from_prompt_tokens(prompt_tokens, steps, &mut cpu_scratch)?;
    let sequence = run_device_sequence(model, prompt_tokens, steps)?;
    let mut counters = CudaDecodeCounters::default();
    counters.record_sequence(&sequence);
    if sequence.status != SmokeStatus::Ok {
        return Err(NervaError::InvalidArgument {
            reason: sequence
                .error
                .unwrap_or_else(|| "CUDA HF decode sequence failed".to_string()),
        });
    }

    let tokens = sequence
        .tokens
        .iter()
        .copied()
        .map(TokenId)
        .collect::<Vec<_>>();
    let ledgers = sequence_ledgers(&sequence);
    for ledger in &ledgers {
        ledger.require_zero_hot_path_allocations()?;
    }
    let expected_tokens = output.generated_tokens;
    let cpu_ledgers = output.ledgers;
    let status = if tokens == expected_tokens {
        SmokeStatus::Ok
    } else {
        SmokeStatus::Failed
    };
    let error = (status != SmokeStatus::Ok)
        .then(|| "CUDA HF device sequence tokens did not match CPU reference".to_string());

    Ok(build_summary(
        status,
        DecodeParts::new(steps, tokens, expected_tokens, ledgers),
        &cpu_ledgers,
        counters,
        error,
    ))
}

pub fn run_loaded_hf_causal_lm_cuda_prompt_decode(
    runtime: &Runtime,
    loaded: &HfCausalLmLoaded,
    prompt_tokens: &[TokenId],
    steps: usize,
    compute_capability: Option<u32>,
) -> Result<HfCudaSeedDecodeSummary> {
    let resident_weights = loaded_resident_weight_summary(runtime, loaded, compute_capability)?;
    let mut summary = run_hf_causal_lm_cuda_prompt_decode(&loaded.model, prompt_tokens, steps)?;
    summary.hot_path_allocations += resident_weights.hot_path_allocations;
    summary.resident_weights = resident_weights;
    Ok(summary)
}

pub fn run_loaded_hf_causal_lm_cuda_seed_decode(
    runtime: &Runtime,
    loaded: &HfCausalLmLoaded,
    seed: TokenId,
    steps: usize,
    compute_capability: Option<u32>,
) -> Result<HfCudaSeedDecodeSummary> {
    run_loaded_hf_causal_lm_cuda_prompt_decode(runtime, loaded, &[seed], steps, compute_capability)
}
