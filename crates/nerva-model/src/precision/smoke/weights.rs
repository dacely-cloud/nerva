#[derive(Clone, Debug)]
pub(crate) struct SmokeWeights {
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

pub(crate) fn smoke_weights() -> SmokeWeights {
    SmokeWeights {
        rms_attn_weight: vec![1.0, 1.0],
        rms_mlp_weight: vec![1.0, 1.0],
        w_q: vec![1.0, 0.0, 0.0, 1.0],
        w_k: vec![1.0, 0.0, 0.0, 1.0],
        w_v: vec![1.0, 0.0, 0.0, 1.0],
        w_o: vec![1.0, 0.0, 0.0, 1.0],
        w_gate: vec![0.5, 0.0, 0.0, 0.5],
        w_up: vec![1.0, 0.0, 0.0, 1.0],
        w_down: vec![1.0, 0.0, 0.0, 1.0],
        rms_eps: 1e-5,
    }
}
