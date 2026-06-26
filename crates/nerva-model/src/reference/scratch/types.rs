use crate::common::shape::TransformerBlockShape;

#[derive(Clone, Debug)]
pub struct TransformerBlockScratch {
    pub(crate) shape: TransformerBlockShape,
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) mlp_norm: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) attn: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) ff: Vec<f32>,
    pub(crate) down: Vec<f32>,
}
