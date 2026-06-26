use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;
use nerva_core::types::id::transaction::TransactionId;

use nerva_ledger::types::token::ledger::TokenLedger;

use crate::graph::layout::GraphLayout;
use crate::token::ring::{DeviceTokenRef, TokenInputSource};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SyntheticStepPlan {
    pub transaction_id: TransactionId,
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub token_index: u64,
    pub input_token: TokenId,
    pub input_source: TokenInputSource,
    pub layout: GraphLayout,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StepOutput {
    pub request_id: RequestId,
    pub sequence_id: SequenceId,
    pub token_index: u64,
    pub input_source: TokenInputSource,
    pub device_token_ref: DeviceTokenRef,
    pub token: TokenId,
    pub finished: bool,
    pub ledger: TokenLedger,
}
