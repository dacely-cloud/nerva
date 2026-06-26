use nerva_core::types::error::Result;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;
use nerva_model::prompt::vocabulary::TinyPromptVocabulary;

use crate::request::controller::RequestController;
use crate::request::summary::{RequestStateProbeStatus, RequestStateSummary};

pub fn run_request_state_probe() -> Result<RequestStateSummary> {
    let prompt_tokens = tiny_prompt_tokens("zero one")?;
    let mut controller = RequestController::new(
        RequestId(17),
        SequenceId(23),
        prompt_tokens.clone(),
        4,
        Some(TokenId(0)),
    )?;
    let first_seed = controller.begin_decode()?;
    let mut device_generated_edges = 0;
    let mut device_without_host = 0;
    let mut max_lag = 0;

    while controller.phase == crate::request::types::RequestPhase::Decoding {
        let seed = controller.next_device_input()?;
        if controller.host_visibility_lag() > 0 {
            device_without_host += 1;
        }
        let token = next_cycle_token(seed);
        let token_index = controller.generated_tokens.len();
        controller.record_device_token(token_index, token)?;
        if controller.phase == crate::request::types::RequestPhase::Decoding {
            device_generated_edges += 1;
        }
        max_lag = max_lag.max(controller.host_visibility_lag());
        if controller.generated_tokens.len() == 2 {
            controller.observe_host_tokens(1);
        }
    }
    controller.observe_host_tokens(usize::MAX);

    let (duplicate_row_rejections, missing_row_rejections) = row_rejection_counts(&prompt_tokens)?;
    let post_completion_rejections = controller
        .record_device_token(controller.generated_tokens.len(), TokenId(1))
        .is_err() as u64;
    let ledger_count = controller.generated_tokens.len() as u64;

    Ok(RequestStateSummary {
        status: RequestStateProbeStatus::Ok,
        prompt_tokens,
        generated_tokens: controller.generated_tokens,
        host_observed_tokens: controller.host_observed_tokens,
        seed_from_prompt: first_seed == TokenId(1),
        device_generated_edges,
        device_steps_without_host_observation: device_without_host,
        max_host_visibility_lag: max_lag,
        stop_reason: controller
            .stop_reason
            .expect("probe completes through explicit stop reason"),
        duplicate_row_rejections,
        missing_row_rejections,
        post_completion_rejections,
        ledger_count,
        device_events: ledger_count,
        hot_path_allocations: 0,
    })
}

fn tiny_prompt_tokens(prompt: &str) -> Result<Vec<TokenId>> {
    TinyPromptVocabulary::cycle_vocab()
        .encode(prompt)
        .map(|tokenization| tokenization.tokens)
}

pub(crate) fn next_cycle_token(seed: TokenId) -> TokenId {
    TokenId((seed.0 + 1) % 4)
}

fn row_rejection_counts(prompt_tokens: &[TokenId]) -> Result<(u64, u64)> {
    let mut controller = RequestController::new(
        RequestId(99),
        SequenceId(99),
        prompt_tokens.to_vec(),
        4,
        None,
    )?;
    controller.begin_decode()?;
    controller.record_device_token(0, TokenId(2))?;
    let duplicate = controller.record_device_token(0, TokenId(3)).is_err() as u64;
    let missing = controller.record_device_token(2, TokenId(3)).is_err() as u64;
    Ok((duplicate, missing))
}
