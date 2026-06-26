use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use crate::request::types::{HostObservationBatch, RequestPhase, StopReason};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestController {
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub prompt_tokens: Vec<TokenId>,
    pub generated_tokens: Vec<TokenId>,
    pub host_observed_tokens: Vec<TokenId>,
    pub phase: RequestPhase,
    pub max_new_tokens: usize,
    pub eos_token: Option<TokenId>,
    pub stop_reason: Option<StopReason>,
    host_cursor: usize,
}

impl RequestController {
    pub fn new(
        request_id: RequestId,
        sequence_id: SequenceId,
        prompt_tokens: Vec<TokenId>,
        max_new_tokens: usize,
        eos_token: Option<TokenId>,
    ) -> Result<Self> {
        if prompt_tokens.is_empty() || max_new_tokens == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "request requires prompt tokens and non-zero max_new_tokens".to_string(),
            });
        }
        Ok(Self {
            request_id,
            sequence_id,
            prompt_tokens,
            generated_tokens: Vec::with_capacity(max_new_tokens),
            host_observed_tokens: Vec::with_capacity(max_new_tokens),
            phase: RequestPhase::PromptReady,
            max_new_tokens,
            eos_token,
            stop_reason: None,
            host_cursor: 0,
        })
    }

    pub fn begin_decode(&mut self) -> Result<TokenId> {
        if self.phase != RequestPhase::PromptReady {
            return Err(NervaError::InvalidArgument {
                reason: "request decode already started".to_string(),
            });
        }
        self.phase = RequestPhase::Decoding;
        self.next_device_input()
    }

    pub fn next_device_input(&self) -> Result<TokenId> {
        if self.phase == RequestPhase::Completed {
            return Err(NervaError::InvalidArgument {
                reason: "completed request has no next device input".to_string(),
            });
        }
        Ok(self
            .generated_tokens
            .last()
            .copied()
            .unwrap_or_else(|| *self.prompt_tokens.last().expect("validated prompt")))
    }

    pub fn record_device_token(&mut self, token_index: usize, token: TokenId) -> Result<()> {
        if self.phase != RequestPhase::Decoding {
            return Err(NervaError::InvalidArgument {
                reason: "device token publication requires decoding phase".to_string(),
            });
        }
        if token_index != self.generated_tokens.len() {
            return Err(NervaError::InvalidArgument {
                reason: format!(
                    "token row {token_index} does not match next row {}",
                    self.generated_tokens.len()
                ),
            });
        }
        self.generated_tokens.push(token);
        if self.eos_token == Some(token) {
            self.complete(StopReason::EosToken);
        } else if self.generated_tokens.len() == self.max_new_tokens {
            self.complete(StopReason::MaxNewTokens);
        }
        Ok(())
    }

    pub fn observe_host_tokens(&mut self, max_tokens: usize) -> HostObservationBatch {
        let start_index = self.host_cursor;
        let end = self
            .generated_tokens
            .len()
            .min(self.host_cursor.saturating_add(max_tokens));
        let tokens = self.generated_tokens[start_index..end].to_vec();
        self.host_observed_tokens.extend_from_slice(&tokens);
        self.host_cursor = end;
        HostObservationBatch {
            start_index,
            tokens,
        }
    }

    pub fn host_visibility_lag(&self) -> usize {
        self.generated_tokens.len().saturating_sub(self.host_cursor)
    }

    fn complete(&mut self, reason: StopReason) {
        self.phase = RequestPhase::Completed;
        self.stop_reason = Some(reason);
    }
}
