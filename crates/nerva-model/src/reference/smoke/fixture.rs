use nerva_core::types::error::Result;

use crate::common::shape::TransformerBlockShape;
use crate::reference::block::types::ReferenceTransformerBlock;

pub(crate) fn reference_smoke_block() -> Result<ReferenceTransformerBlock> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    ReferenceTransformerBlock::new(
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
    )
}
