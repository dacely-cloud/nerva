use nerva_core::types::id::token::TokenId;

const MAX_NGRAM_ORDER: usize = 4;

pub(super) fn draft_ngram_block(
    prompt_tokens: &[TokenId],
    generated_tokens: &[TokenId],
    draft_tokens: usize,
    vocab_size: usize,
) -> Vec<u32> {
    let mut history = prompt_tokens
        .iter()
        .chain(generated_tokens.iter())
        .map(|token| token.0)
        .collect::<Vec<_>>();
    let mut draft = Vec::with_capacity(draft_tokens);
    for _ in 0..draft_tokens {
        let next = predict_next_token(&history, vocab_size).unwrap_or(0);
        draft.push(next);
        history.push(next);
    }
    draft
}

fn predict_next_token(history: &[u32], vocab_size: usize) -> Option<u32> {
    if history.is_empty() || vocab_size == 0 {
        return None;
    }
    let max_order = MAX_NGRAM_ORDER.min(history.len());
    for order in (1..=max_order).rev() {
        let suffix = &history[history.len() - order..];
        for start in (0..history.len().saturating_sub(order)).rev() {
            if &history[start..start + order] == suffix {
                let candidate = history[start + order];
                if (candidate as usize) < vocab_size {
                    return Some(candidate);
                }
            }
        }
    }
    history
        .last()
        .copied()
        .filter(|token| (*token as usize) < vocab_size)
}
