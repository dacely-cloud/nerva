use nerva_core::types::error::Result;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::common::hash::hash_f32s;
use crate::common::shape::TransformerBlockShape;
use crate::reference::block::ReferenceTransformerBlock;
use crate::reference::scratch::TransformerBlockScratch;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReferenceBlockSmokeStatus {
    Ok,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReferenceBlockSmokeSummary {
    pub status: ReferenceBlockSmokeStatus,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub output: [f32; 2],
    pub output_hash: u64,
    pub hot_path_allocations: u64,
}

impl ReferenceBlockSmokeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            ReferenceBlockSmokeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"hidden\":{},\"heads\":{},\"intermediate\":{},\"output\":[{},{}],\"output_hash\":{},\"hot_path_allocations\":{}}}",
            status,
            self.hidden,
            self.heads,
            self.intermediate,
            self.output[0],
            self.output[1],
            self.output_hash,
            self.hot_path_allocations,
        )
    }
}

pub fn reference_block_smoke() -> Result<ReferenceBlockSmokeSummary> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let block = ReferenceTransformerBlock::new(
        shape,
        vec![1.0, 1.0],
        vec![1.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.5, 0.0, 0.0, 0.5],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        1e-5,
    )?;
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
