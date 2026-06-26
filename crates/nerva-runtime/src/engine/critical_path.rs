use nerva_core::types::error::Result;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;
use nerva_core::types::id::token::TokenId;

use nerva_ledger::types::token::critical::TokenCriticalPathReport;

use crate::engine::runtime::Runtime;

impl Runtime {
    pub fn run_critical_path_probe(&self) -> Result<TokenCriticalPathReport> {
        let mut engine = self.synthetic_engine(4)?;
        let output = engine
            .launch_device_next(RequestId(1), SequenceId(1), 0, TokenId(1))?
            .collect()?;
        output.ledger.require_classified_syncs()?;
        output.ledger.require_zero_hot_path_allocations()?;
        TokenCriticalPathReport::from_ledger(&output.ledger, self.config.device)
    }
}
