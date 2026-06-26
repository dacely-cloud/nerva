use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::hash::hash_tokens;
use crate::prompt::summary::{TinyPromptDecodeStatus, TinyPromptDecodeSummary};
use crate::prompt::vocabulary::TinyPromptVocabulary;
use crate::tiny::scratch::TinyGreedyDecodeScratch;
use crate::tiny::smoke::tiny_cycle_model;

pub fn tiny_prompt_decode_smoke(prompt: &str, steps: usize) -> Result<TinyPromptDecodeSummary> {
    if steps == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "prompt decode steps must be non-zero".to_string(),
        });
    }

    let vocabulary = TinyPromptVocabulary::cycle_vocab();
    let tokenization = vocabulary.encode(prompt)?;
    let seed_token = *tokenization
        .tokens
        .last()
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "prompt must contain a final seed token".to_string(),
        })?;

    let model = tiny_cycle_model()?;
    let mut scratch = TinyGreedyDecodeScratch::new(model.shape(), model.vocab_size())?;
    let output = model.decode_greedy(seed_token, steps, &mut scratch)?;
    let prompt_text_roundtrip = vocabulary.decode(&tokenization.tokens)?;
    let generated_text = vocabulary.decode(&output.tokens)?;

    let mut full_sequence = tokenization.tokens.clone();
    full_sequence.extend_from_slice(&output.tokens);
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

    Ok(TinyPromptDecodeSummary {
        status: TinyPromptDecodeStatus::Ok,
        prompt: tokenization.original_text,
        prompt_tokens: tokenization.tokens,
        seed_token,
        steps,
        vocabulary_covered: vocabulary.contains_all(&full_sequence),
        output_hash: hash_tokens(&full_sequence),
        generated_tokens: output.tokens,
        full_sequence,
        generated_text,
        prompt_text_roundtrip,
        seed_from_prompt: true,
        ledger_count: output.ledgers.len() as u64,
        device_events,
        total_latency_ns,
        hot_path_allocations,
    })
}
