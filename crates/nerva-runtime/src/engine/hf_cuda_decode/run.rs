use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::causal_lm::types::{HfCausalLmDecodeScratch, HfCausalLmModel};

use crate::engine::hf_cuda_decode::sequence::run_device_sequence;
use crate::engine::hf_cuda_decode::sequence_ledger::sequence_ledgers;
use crate::engine::hf_cuda_decode::summary::HfCudaSeedDecodeSummary;
use crate::engine::hf_cuda_decode::totals::{CudaDecodeCounters, DecodeParts, build_summary};

pub fn run_hf_causal_lm_cuda_seed_decode(
    model: &HfCausalLmModel,
    seed: TokenId,
    steps: usize,
) -> Result<HfCudaSeedDecodeSummary> {
    if steps == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "HF CUDA seed decode steps must be non-zero".to_string(),
        });
    }
    let mut cpu_scratch = HfCausalLmDecodeScratch::new(model.shape(), model.metadata().vocab_size)?;
    let (expected_tokens, cpu_ledgers) = model.decode_greedy(seed, steps, &mut cpu_scratch)?;
    let sequence = run_device_sequence(model, seed, steps)?;
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
