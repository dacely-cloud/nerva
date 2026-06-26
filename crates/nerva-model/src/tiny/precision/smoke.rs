use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::TokenId;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::TokenLedger;

use crate::common::hash::hash_tokens;
use crate::common::shape::TransformerBlockShape;
use crate::common::token::expected_cycle;
use crate::precision::block::PrecisionTransformerBlock;
use crate::tiny::precision::model::TinyPrecisionGreedyModel;
use crate::tiny::precision::output::{
    TinyPrecisionGreedyDecodeStatus, TinyPrecisionGreedyDecodeSummary,
};
use crate::tiny::precision::scratch::TinyPrecisionGreedyDecodeScratch;

pub fn tiny_precision_greedy_decode_smoke(
    dtype: DType,
    steps: usize,
) -> Result<TinyPrecisionGreedyDecodeSummary> {
    let model = tiny_precision_cycle_model(dtype)?;
    let seed_token = TokenId(0);
    let mut scratch = TinyPrecisionGreedyDecodeScratch::new(model.shape(), model.vocab_size())?;
    let output = model.decode_greedy(seed_token, steps, &mut scratch)?;
    let expected_tokens = expected_cycle(seed_token, steps, model.vocab_size());
    let parity = output.tokens == expected_tokens;
    if !parity {
        return Err(NervaError::InvalidArgument {
            reason: "tiny precision greedy decode token parity failed".to_string(),
        });
    }
    let hot_path_allocations = output
        .ledgers
        .iter()
        .map(|ledger| ledger.hot_path_allocations)
        .sum();
    let cpu_events = output
        .ledgers
        .iter()
        .map(|ledger| ledger.event_count(LedgerEventKind::CpuActivity))
        .sum();
    let execution_decisions = output
        .ledgers
        .iter()
        .map(|ledger| ledger.execution_decisions.len() as u64)
        .sum();
    let total_latency_ns = output
        .ledgers
        .iter()
        .map(TokenLedger::total_latency_ns)
        .sum();

    let summary = TinyPrecisionGreedyDecodeSummary {
        status: TinyPrecisionGreedyDecodeStatus::Ok,
        dtype: model.dtype(),
        seed_token,
        steps,
        vocab_size: model.vocab_size(),
        output_hash: hash_tokens(&output.tokens),
        tokens: output.tokens,
        expected_tokens,
        parity,
        ledger_count: output.ledgers.len() as u64,
        cpu_events,
        execution_decisions,
        total_latency_ns,
        hot_path_allocations,
    };
    if summary.passed() {
        Ok(summary)
    } else {
        Err(NervaError::InvalidArgument {
            reason: "tiny precision decode ledger invariants failed".to_string(),
        })
    }
}

pub fn tiny_precision_cycle_model(dtype: DType) -> Result<TinyPrecisionGreedyModel> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let block = PrecisionTransformerBlock::new_from_f32(
        dtype,
        shape,
        &[1.0, 1.0],
        &[1.0, 1.0],
        &[0.0; 4],
        &[0.0; 4],
        &[0.0; 4],
        &[0.0; 4],
        &[0.0; 4],
        &[0.0; 4],
        &[0.0; 4],
        1e-5,
    )?;
    TinyPrecisionGreedyModel::new_from_f32(
        dtype,
        block,
        &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
        &[0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
    )
}
