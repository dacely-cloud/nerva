use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::hash::hash_tokens;
use crate::common::shape::TransformerBlockShape;
use crate::common::token::expected_cycle;
use crate::reference::block::types::ReferenceTransformerBlock;
use crate::tiny::model::types::TinyGreedyModel;
use crate::tiny::output::{TinyGreedyDecodeStatus, TinyGreedyDecodeSummary};
use crate::tiny::scratch::TinyGreedyDecodeScratch;

pub fn tiny_greedy_decode_smoke(steps: usize) -> Result<TinyGreedyDecodeSummary> {
    let model = tiny_cycle_model()?;
    let seed_token = TokenId(0);
    let mut scratch = TinyGreedyDecodeScratch::new(model.shape(), model.vocab_size())?;
    let output = model.decode_greedy(seed_token, steps, &mut scratch)?;
    let expected_tokens = expected_cycle(seed_token, steps, model.vocab_size());
    let parity = output.tokens == expected_tokens;
    if !parity {
        return Err(NervaError::InvalidArgument {
            reason: "tiny greedy decode token parity failed".to_string(),
        });
    }
    let hot_path_allocations = output
        .ledgers
        .iter()
        .map(|ledger| ledger.hot_path_allocations)
        .sum();
    let device_events = output
        .ledgers
        .iter()
        .map(|ledger| ledger.event_count(LedgerEventKind::DeviceActivity))
        .sum();
    let total_latency_ns = output
        .ledgers
        .iter()
        .map(TokenLedger::total_latency_ns)
        .sum();

    Ok(TinyGreedyDecodeSummary {
        status: TinyGreedyDecodeStatus::Ok,
        seed_token,
        steps,
        vocab_size: model.vocab_size(),
        output_hash: hash_tokens(&output.tokens),
        tokens: output.tokens,
        expected_tokens,
        parity,
        ledger_count: output.ledgers.len() as u64,
        device_events,
        total_latency_ns,
        hot_path_allocations,
    })
}

pub(crate) fn tiny_cycle_model() -> Result<TinyGreedyModel> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let block = ReferenceTransformerBlock::zero_for_shape(shape)?;
    TinyGreedyModel::new(
        4,
        block,
        vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0],
        vec![0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
    )
}
