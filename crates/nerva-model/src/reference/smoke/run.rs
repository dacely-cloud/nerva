use nerva_core::types::error::Result;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::hash::hash_f32s;
use crate::reference::scratch::types::TransformerBlockScratch;
use crate::reference::smoke::fixture::reference_smoke_block;
use crate::reference::smoke::status::ReferenceBlockSmokeStatus;
use crate::reference::smoke::summary::ReferenceBlockSmokeSummary;

pub fn reference_block_smoke() -> Result<ReferenceBlockSmokeSummary> {
    let block = reference_smoke_block()?;
    let shape = block.shape();
    let input = [1.0, 2.0];
    let mut scratch = TransformerBlockScratch::new(shape)?;
    let mut output = [0.0; 2];
    let mut ledger = TokenLedger::new(0);
    block.forward_into(&input, &mut scratch, &mut output, &mut ledger)?;
    Ok(ReferenceBlockSmokeSummary {
        status: ReferenceBlockSmokeStatus::Ok,
        hidden: shape.hidden,
        heads: shape.heads,
        intermediate: shape.intermediate,
        output,
        output_hash: hash_f32s(&output),
        hot_path_allocations: ledger.hot_path_allocations,
    })
}
