use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptVocabularyEntry {
    pub token: TokenId,
    pub text: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TinyPromptVocabulary {
    entries: Vec<PromptVocabularyEntry>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptTokenization {
    pub original_text: String,
    pub tokens: Vec<TokenId>,
}

impl TinyPromptVocabulary {
    pub fn cycle_vocab() -> Self {
        Self {
            entries: vec![
                PromptVocabularyEntry {
                    token: TokenId(0),
                    text: "zero",
                },
                PromptVocabularyEntry {
                    token: TokenId(1),
                    text: "one",
                },
                PromptVocabularyEntry {
                    token: TokenId(2),
                    text: "two",
                },
                PromptVocabularyEntry {
                    token: TokenId(3),
                    text: "three",
                },
            ],
        }
    }

    pub fn encode(&self, text: &str) -> Result<PromptTokenization> {
        let mut tokens = Vec::new();
        for piece in text.split_whitespace() {
            let normalized = piece.to_ascii_lowercase();
            let token = self
                .entries
                .iter()
                .find(|entry| entry.text == normalized)
                .map(|entry| entry.token)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: format!("prompt token {piece:?} is not in tiny vocabulary"),
                })?;
            tokens.push(token);
        }
        if tokens.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "prompt must contain at least one token".to_string(),
            });
        }
        Ok(PromptTokenization {
            original_text: text.to_string(),
            tokens,
        })
    }

    pub fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        let mut words = Vec::with_capacity(tokens.len());
        for token in tokens {
            let text = self
                .entries
                .iter()
                .find(|entry| entry.token == *token)
                .map(|entry| entry.text)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: format!("token id {} is not in tiny vocabulary", token.0),
                })?;
            words.push(text);
        }
        Ok(words.join(" "))
    }

    pub fn contains_all(&self, tokens: &[TokenId]) -> bool {
        tokens
            .iter()
            .all(|token| self.entries.iter().any(|entry| entry.token == *token))
    }
}
