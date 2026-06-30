use crate::deepseek_mhc::hc_head::{
    CudaDeepSeekMhcHeadInput, CudaDeepSeekMhcHeadSummary, deepseek_mhc_head,
};
use crate::smoke::status::SmokeStatus;

pub fn deepseek_mhc_head_smoke() -> CudaDeepSeekMhcHeadSummary {
    let fixture = deepseek_mhc_head_fixture();
    let summary = deepseek_mhc_head(fixture.input());
    if summary.status != SmokeStatus::Ok {
        return summary;
    }

    let expected = reference_mhc_head(fixture.input());
    let matches_output = summary
        .output
        .iter()
        .zip(expected.iter())
        .all(|(actual, expected)| (actual - expected).abs() <= 1e-5);
    if matches_output
        && summary.output_hash != 0
        && summary.kernel_launches == 1
        && summary.sync_calls == 1
        && summary.hot_path_allocations == 0
    {
        return summary;
    }

    let mut failed = summary;
    failed.status = SmokeStatus::Failed;
    failed.error = Some(format!(
        "CUDA DeepSeek mHC head smoke mismatch: matches_output={} output_hash={} kernel_launches={}",
        matches_output, failed.output_hash, failed.kernel_launches
    ));
    failed
}

#[derive(Clone, Debug)]
pub(crate) struct DeepSeekMhcHeadFixture {
    pub(crate) tokens: u32,
    pub(crate) hc_mult: u32,
    pub(crate) hidden_size: u32,
    pub(crate) rms_eps: f32,
    pub(crate) hc_eps: f32,
    pub(crate) hc_scale: f32,
    pub(crate) hidden_states: Vec<f32>,
    pub(crate) fn_weights: Vec<f32>,
    pub(crate) hc_base: Vec<f32>,
}

impl DeepSeekMhcHeadFixture {
    pub(crate) fn input(&self) -> CudaDeepSeekMhcHeadInput<'_> {
        CudaDeepSeekMhcHeadInput {
            tokens: self.tokens,
            hc_mult: self.hc_mult,
            hidden_size: self.hidden_size,
            rms_eps: self.rms_eps,
            hc_eps: self.hc_eps,
            hc_scale: self.hc_scale,
            hidden_states: &self.hidden_states,
            fn_weights: &self.fn_weights,
            hc_base: &self.hc_base,
        }
    }
}

pub(crate) fn deepseek_mhc_head_fixture() -> DeepSeekMhcHeadFixture {
    DeepSeekMhcHeadFixture {
        tokens: 2,
        hc_mult: 2,
        hidden_size: 3,
        rms_eps: 1e-5,
        hc_eps: 0.001,
        hc_scale: 1.25,
        hidden_states: vec![
            0.5, -1.0, 1.5, 0.25, 0.75, -0.5, -0.2, 0.4, 0.8, 1.2, -0.6, 0.3,
        ],
        fn_weights: vec![
            0.3, -0.2, 0.1, 0.4, -0.5, 0.2, -0.1, 0.6, 0.25, -0.35, 0.15, 0.45,
        ],
        hc_base: vec![0.05, -0.1],
    }
}

pub(crate) fn reference_mhc_head(input: CudaDeepSeekMhcHeadInput<'_>) -> Vec<f32> {
    let tokens = input.tokens as usize;
    let hc_mult = input.hc_mult as usize;
    let hidden_size = input.hidden_size as usize;
    let hc_hidden_size = hc_mult * hidden_size;
    let mut output = vec![0.0f32; tokens * hidden_size];
    for token in 0..tokens {
        let token_start = token * hc_hidden_size;
        let token_values = &input.hidden_states[token_start..token_start + hc_hidden_size];
        let sqrsum = token_values.iter().map(|value| value * value).sum::<f32>();
        let rms_scale = 1.0 / (sqrsum / hc_hidden_size as f32 + input.rms_eps).sqrt();
        let mut gates = vec![0.0f32; hc_mult];
        for (channel, gate) in gates.iter_mut().enumerate() {
            let row_start = channel * hc_hidden_size;
            let row = &input.fn_weights[row_start..row_start + hc_hidden_size];
            let mix = row
                .iter()
                .zip(token_values.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
                * rms_scale;
            *gate = sigmoid(mix * input.hc_scale + input.hc_base[channel]) + input.hc_eps;
        }
        for hidden in 0..hidden_size {
            let mut value = 0.0f32;
            for channel in 0..hc_mult {
                value += gates[channel] * token_values[channel * hidden_size + hidden];
            }
            output[token * hidden_size + hidden] = value;
        }
    }
    output
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}
