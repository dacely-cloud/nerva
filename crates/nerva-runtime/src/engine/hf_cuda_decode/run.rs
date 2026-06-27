use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::sampler::hf_head::request::{
    CUDA_HF_SAMPLER_DTYPE_BF16, CUDA_HF_SAMPLER_DTYPE_F16, CudaHfSamplerRequest,
};
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_model::causal_lm::types::{HfCausalLmDecodeScratch, HfCausalLmModel};

use crate::engine::cuda_block::run_precision_block_on_cuda;
use crate::engine::hf_cuda_decode::ledger::{record_layer_execution, record_sampler_execution};
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
    let mut sample_scratch =
        HfCausalLmDecodeScratch::new(model.shape(), model.metadata().vocab_size)?;
    let mut counters = CudaDecodeCounters::default();
    let mut ledgers = Vec::with_capacity(steps);
    let mut tokens = Vec::with_capacity(steps);
    let mut current = seed;

    for step in 0..steps {
        let mut hidden = model.embedding_row(current)?.to_vec();
        let mut ledger = TokenLedger::new(step as u64);
        for layer_index in 0..model.layer_count() {
            let layer = model
                .layer(layer_index)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: format!("HF CUDA layer index {layer_index} is out of range"),
                })?;
            let cuda = run_precision_block_on_cuda(layer, &hidden, step as u32)?;
            counters.record_block(&cuda);
            record_layer_execution(&mut ledger, &cuda);
            if cuda.status != SmokeStatus::Ok {
                ledgers.push(ledger);
                return Ok(build_summary(
                    cuda.status,
                    DecodeParts::new(steps, tokens, expected_tokens, ledgers),
                    &cpu_ledgers,
                    counters,
                    cuda.error,
                ));
            }
            hidden = cuda.output;
        }
        let sampler = CudaHfSamplerRequest {
            dtype: cuda_dtype(model.dtype())?,
            hidden: model.metadata().hidden_size,
            vocab_size: model.metadata().vocab_size,
            token_index: step as u64,
            rms_eps: model.rms_eps(),
            hidden_bits: &hidden,
            final_norm_weight: model.final_norm_weight(),
            lm_head: model.lm_head(),
        }
        .run();
        counters.record_sampler(&sampler);
        record_sampler_execution(&mut ledger, &sampler);
        if sampler.status != SmokeStatus::Ok {
            ledgers.push(ledger);
            return Ok(build_summary(
                sampler.status,
                DecodeParts::new(steps, tokens, expected_tokens, ledgers),
                &cpu_ledgers,
                counters,
                sampler.error,
            ));
        }
        let token = TokenId(sampler.token);
        let expected = model.sample_encoded_hidden(&hidden, &mut sample_scratch)?;
        if token != expected {
            ledgers.push(ledger);
            tokens.push(token);
            return Ok(build_summary(
                SmokeStatus::Failed,
                DecodeParts::new(steps, tokens, expected_tokens, ledgers),
                &cpu_ledgers,
                counters,
                Some("CUDA HF sampler token did not match CPU reference".to_string()),
            ));
        }
        ledger.require_zero_hot_path_allocations()?;
        tokens.push(token);
        ledgers.push(ledger);
        if model.metadata().eos_token_id == Some(token.0) {
            break;
        }
        current = token;
    }

    Ok(build_summary(
        SmokeStatus::Ok,
        DecodeParts::new(steps, tokens, expected_tokens, ledgers),
        &cpu_ledgers,
        counters,
        None,
    ))
}

fn cuda_dtype(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(CUDA_HF_SAMPLER_DTYPE_F16),
        DType::BF16 => Ok(CUDA_HF_SAMPLER_DTYPE_BF16),
        other => Err(NervaError::InvalidArgument {
            reason: format!("CUDA HF sampler does not support dtype {other:?}"),
        }),
    }
}
