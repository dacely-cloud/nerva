use nerva_core::types::id::token::TokenId;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RequestPhase {
    PromptReady,
    Decoding,
    Completed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StopReason {
    MaxNewTokens,
    EosToken,
}

impl StopReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MaxNewTokens => "max_new_tokens",
            Self::EosToken => "eos_token",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostObservationBatch {
    pub start_index: usize,
    pub tokens: Vec<TokenId>,
}
