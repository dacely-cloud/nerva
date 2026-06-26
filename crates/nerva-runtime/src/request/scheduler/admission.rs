use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestAdmission {
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub prompt_tokens: Vec<TokenId>,
    pub max_new_tokens: usize,
    pub eos_token: Option<TokenId>,
}
