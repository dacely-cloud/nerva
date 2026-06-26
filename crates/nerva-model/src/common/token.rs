use nerva_core::types::{NervaError, Result, TokenId};

use crate::common::validate::require_len;

pub(crate) fn require_token_in_vocab(token: TokenId, vocab_size: usize) -> Result<()> {
    if token.0 as usize >= vocab_size {
        Err(NervaError::InvalidArgument {
            reason: format!(
                "token id {} is outside tiny model vocabulary {}",
                token.0, vocab_size
            ),
        })
    } else {
        Ok(())
    }
}

pub(crate) fn copy_embedding_row(
    embeddings: &[f32],
    hidden: usize,
    token: TokenId,
    output: &mut [f32],
) -> Result<()> {
    require_token_in_vocab(token, embeddings.len() / hidden)?;
    require_len("embedding output", output.len(), hidden)?;
    let start = token.0 as usize * hidden;
    let end = start + hidden;
    output.copy_from_slice(&embeddings[start..end]);
    Ok(())
}

pub(crate) fn greedy_argmax(logits: &[f32]) -> Result<TokenId> {
    if logits.is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: "greedy argmax requires non-empty logits".to_string(),
        });
    }
    let mut best_index = 0usize;
    let mut best_value = logits[0];
    if !best_value.is_finite() {
        return Err(NervaError::InvalidArgument {
            reason: "greedy argmax saw non-finite logit".to_string(),
        });
    }
    for (index, value) in logits.iter().copied().enumerate().skip(1) {
        if !value.is_finite() {
            return Err(NervaError::InvalidArgument {
                reason: "greedy argmax saw non-finite logit".to_string(),
            });
        }
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    Ok(TokenId(best_index as u32))
}

pub(crate) fn expected_cycle(seed_token: TokenId, steps: usize, vocab_size: usize) -> Vec<TokenId> {
    (0..steps)
        .map(|step| TokenId((seed_token.0 + step as u32 + 1) % vocab_size as u32))
        .collect()
}

pub(crate) fn token_ids_to_json(tokens: &[TokenId]) -> String {
    let mut out = String::from("[");
    for (index, token) in tokens.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&token.0.to_string());
    }
    out.push(']');
    out
}
