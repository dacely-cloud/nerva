use crate::common::shape::TransformerBlockShape;

#[derive(Clone, Debug)]
pub struct ReferenceTransformerBlock {
    pub(crate) shape: TransformerBlockShape,
    pub(crate) rms_attn_weight: Vec<f32>,
    pub(crate) rms_mlp_weight: Vec<f32>,
    pub(crate) w_q: Vec<f32>,
    pub(crate) w_k: Vec<f32>,
    pub(crate) w_v: Vec<f32>,
    pub(crate) w_o: Vec<f32>,
    pub(crate) w_gate: Vec<f32>,
    pub(crate) w_up: Vec<f32>,
    pub(crate) w_down: Vec<f32>,
    pub(crate) rms_eps: f32,
}
