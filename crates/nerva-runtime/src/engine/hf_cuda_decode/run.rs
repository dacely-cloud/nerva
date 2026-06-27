use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use nerva_cuda::decode::hf_sequence::weight_plan::{
    CudaHfDecodeSequenceWeightBlock, CudaHfDecodeSequenceWeightPlan,
};
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::causal_lm::types::{HfCausalLmDecodeScratch, HfCausalLmLoaded, HfCausalLmModel};

use crate::engine::hf_cuda_decode::contract::{attach_cuda_weight_contract, cuda_weight_plan};
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
    Ok(
        run_hf_causal_lm_cuda_prompt_decode_with_plan(
            model,
            prompt_tokens,
            steps,
            None,
            &[],
            true,
        )?
        .0,
    )
}

fn run_hf_causal_lm_cuda_prompt_decode_with_plan(
    model: &HfCausalLmModel,
    prompt_tokens: &[TokenId],
    steps: usize,
    weight_plan: Option<CudaHfDecodeSequenceWeightPlan>,
    weight_blocks: &[CudaHfDecodeSequenceWeightBlock],
    verify_reference: bool,
) -> Result<(HfCudaSeedDecodeSummary, CudaHfDecodeSequenceSummary)> {
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
    let cpu_output = if verify_reference {
        let mut cpu_scratch = HfCausalLmDecodeScratch::new_with_context(
            model.shape(),
            model.metadata().vocab_size,
            model.layer_count(),
            context_tokens,
        )?;
        Some(model.decode_greedy_from_prompt_tokens(prompt_tokens, steps, &mut cpu_scratch)?)
    } else {
        None
    };
    let sequence = run_device_sequence(model, prompt_tokens, steps, weight_plan, weight_blocks)?;
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
    let expected_tokens = cpu_output
        .as_ref()
        .map(|output| output.generated_tokens.clone())
        .unwrap_or_default();
    let cpu_ledgers = cpu_output
        .as_ref()
        .map(|output| output.ledgers.as_slice())
        .unwrap_or(&[]);
    let status = if !verify_reference || tokens == expected_tokens {
        SmokeStatus::Ok
    } else {
        SmokeStatus::Failed
    };
    let error = (verify_reference && status != SmokeStatus::Ok)
        .then(|| "CUDA HF device sequence tokens did not match CPU reference".to_string());
    let reference_mode = if verify_reference {
        "cpu_reference"
    } else {
        "device_only_unverified"
    };

    let summary = build_summary(
        status,
        DecodeParts::new(
            steps,
            tokens,
            expected_tokens,
            reference_mode,
            verify_reference,
            ledgers,
        ),
        cpu_ledgers,
        counters,
        error,
    );
    Ok((summary, sequence))
}

pub fn run_loaded_hf_causal_lm_cuda_prompt_decode(
    runtime: &Runtime,
    loaded: &HfCausalLmLoaded,
    prompt_tokens: &[TokenId],
    steps: usize,
    compute_capability: Option<u32>,
) -> Result<HfCudaSeedDecodeSummary> {
    let resident_weights = loaded_resident_weight_summary(runtime, loaded, compute_capability)?;
    let weight_plan = cuda_weight_plan(&resident_weights.summary, &resident_weights.descriptors)?;
    let (mut summary, sequence) = run_hf_causal_lm_cuda_prompt_decode_with_plan(
        &loaded.model,
        prompt_tokens,
        steps,
        Some(weight_plan),
        &resident_weights.descriptors,
        true,
    )?;
    let mut resident_summary = resident_weights.summary;
    attach_cuda_weight_contract(&mut resident_summary, &sequence)?;
    summary.hot_path_allocations += resident_summary.hot_path_allocations;
    summary.resident_weights = resident_summary;
    Ok(summary)
}

pub fn run_loaded_hf_causal_lm_cuda_prompt_decode_device_only(
    runtime: &Runtime,
    loaded: &HfCausalLmLoaded,
    prompt_tokens: &[TokenId],
    steps: usize,
    compute_capability: Option<u32>,
) -> Result<HfCudaSeedDecodeSummary> {
    let resident_weights = loaded_resident_weight_summary(runtime, loaded, compute_capability)?;
    let weight_plan = cuda_weight_plan(&resident_weights.summary, &resident_weights.descriptors)?;
    let (mut summary, sequence) = run_hf_causal_lm_cuda_prompt_decode_with_plan(
        &loaded.model,
        prompt_tokens,
        steps,
        Some(weight_plan),
        &resident_weights.descriptors,
        false,
    )?;
    let mut resident_summary = resident_weights.summary;
    attach_cuda_weight_contract(&mut resident_summary, &sequence)?;
    summary.hot_path_allocations += resident_summary.hot_path_allocations;
    summary.resident_weights = resident_summary;
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
