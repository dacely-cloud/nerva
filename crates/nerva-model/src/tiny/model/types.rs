use crate::common::shape::TransformerBlockShape;
use crate::reference::block::types::ReferenceTransformerBlock;

#[derive(Clone, Debug)]
pub struct TinyGreedyModel {
    pub(crate) vocab_size: usize,
    pub(crate) shape: TransformerBlockShape,
    pub(crate) block: ReferenceTransformerBlock,
    pub(crate) embeddings: Vec<f32>,
    pub(crate) lm_head: Vec<f32>,
}
