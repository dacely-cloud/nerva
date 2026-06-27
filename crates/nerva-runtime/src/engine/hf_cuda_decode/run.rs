use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_cuda::block::forward::summary::CudaBlockForwardSummary;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;
use nerva_model::causal_lm::types::{HfCausalLmDecodeScratch, HfCausalLmModel};

use crate::engine::cuda_block::run_precision_block_on_cuda;
use crate::engine::hf_cuda_decode::hash::hash_tokens;
use crate::engine::hf_cuda_decode::ledger::record_layer_execution;
use crate::engine::hf_cuda_decode::summary::HfCudaSeedDecodeSummary;

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
            counters.record_cuda(&cuda);
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
        let token = model.sample_encoded_hidden(&hidden, &mut sample_scratch)?;
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

#[derive(Default)]
struct CudaDecodeCounters {
    resident_weight_bytes: u64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    kernel_launches: u64,
    sync_calls: u64,
    hot_path_allocations: u64,
}

impl CudaDecodeCounters {
    fn record_cuda(&mut self, cuda: &CudaBlockForwardSummary) {
        self.resident_weight_bytes += cuda.resident_weight_bytes;
        self.h2d_bytes += cuda.h2d_bytes;
        self.d2h_bytes += cuda.d2h_bytes;
        self.kernel_launches += cuda.kernel_launches;
        self.sync_calls += cuda.sync_calls;
        self.hot_path_allocations += cuda.hot_path_allocations;
    }
}

struct DecodeParts {
    steps_requested: usize,
    tokens: Vec<TokenId>,
    expected_tokens: Vec<TokenId>,
    ledgers: Vec<TokenLedger>,
}

impl DecodeParts {
    fn new(
        steps_requested: usize,
        tokens: Vec<TokenId>,
        expected_tokens: Vec<TokenId>,
        ledgers: Vec<TokenLedger>,
    ) -> Self {
        Self {
            steps_requested,
            tokens,
            expected_tokens,
            ledgers,
        }
    }
}

fn build_summary(
    status: SmokeStatus,
    parts: DecodeParts,
    cpu_ledgers: &[TokenLedger],
    counters: CudaDecodeCounters,
    error: Option<String>,
) -> HfCudaSeedDecodeSummary {
    let output_hash = hash_tokens(&parts.tokens);
    let expected_hash = hash_tokens(&parts.expected_tokens);
    HfCudaSeedDecodeSummary {
        status,
        steps_requested: parts.steps_requested,
        parity: parts.tokens == parts.expected_tokens,
        ledger_count: parts.ledgers.len() as u64,
        device_events: event_count(&parts.ledgers, LedgerEventKind::DeviceActivity),
        copy_events: event_count(&parts.ledgers, LedgerEventKind::Copy),
        hard_syncs: sync_count(&parts.ledgers, SyncClass::HardSync),
        execution_decisions: execution_decisions(&parts.ledgers),
        resident_weight_bytes: counters.resident_weight_bytes,
        h2d_bytes: counters.h2d_bytes,
        d2h_bytes: counters.d2h_bytes,
        kernel_launches: counters.kernel_launches,
        sync_calls: counters.sync_calls,
        hot_path_allocations: counters.hot_path_allocations
            + hot_path_allocations(&parts.ledgers)
            + hot_path_allocations(cpu_ledgers),
        output_hash,
        expected_hash,
        tokens: parts.tokens,
        expected_tokens: parts.expected_tokens,
        error,
    }
}

fn event_count(ledgers: &[TokenLedger], kind: LedgerEventKind) -> u64 {
    ledgers.iter().map(|ledger| ledger.event_count(kind)).sum()
}

fn sync_count(ledgers: &[TokenLedger], class: SyncClass) -> u64 {
    ledgers
        .iter()
        .map(|ledger| ledger.sync_count_for(class))
        .sum()
}

fn execution_decisions(ledgers: &[TokenLedger]) -> u64 {
    ledgers
        .iter()
        .map(|ledger| ledger.execution_decisions.len() as u64)
        .sum()
}

fn hot_path_allocations(ledgers: &[TokenLedger]) -> u64 {
    ledgers
        .iter()
        .map(|ledger| ledger.hot_path_allocations)
        .sum()
}
