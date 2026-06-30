use crate::deepseek_mhc::hc_head::{
    CudaDeepSeekMhcHeadInput, CudaDeepSeekMhcHeadSummary, deepseek_mhc_head,
};
use crate::deepseek_mhc::post::{
    CudaDeepSeekMhcPostInput, CudaDeepSeekMhcPostSummary, deepseek_mhc_post,
};
use crate::deepseek_mhc::pre::{
    CudaDeepSeekMhcPreInput, CudaDeepSeekMhcPreSummary, deepseek_mhc_pre,
};
use crate::smoke::status::SmokeStatus;

pub fn deepseek_mhc_pre_smoke() -> CudaDeepSeekMhcPreSummary {
    let fixture = deepseek_mhc_pre_fixture();
    let summary = deepseek_mhc_pre(fixture.input());
    if summary.status != SmokeStatus::Ok {
        return summary;
    }

    let expected = reference_mhc_pre(fixture.input());
    let matches_post = close_vec(&summary.post_mix, &expected.post_mix, 1e-5);
    let matches_comb = close_vec(&summary.comb_mix, &expected.comb_mix, 1e-5);
    let matches_layer = close_vec(&summary.layer_input, &expected.layer_input, 1e-5);
    if matches_post
        && matches_comb
        && matches_layer
        && summary.post_mix_hash != 0
        && summary.comb_mix_hash != 0
        && summary.layer_input_hash != 0
        && summary.kernel_launches == 1
        && summary.sync_calls == 1
        && summary.hot_path_allocations == 0
    {
        return summary;
    }

    let mut failed = summary;
    failed.status = SmokeStatus::Failed;
    failed.error = Some(format!(
        "CUDA DeepSeek mHC pre smoke mismatch: post={} comb={} layer={} hashes=({},{},{})",
        matches_post,
        matches_comb,
        matches_layer,
        failed.post_mix_hash,
        failed.comb_mix_hash,
        failed.layer_input_hash
    ));
    failed
}

pub fn deepseek_mhc_post_smoke() -> CudaDeepSeekMhcPostSummary {
    let fixture = deepseek_mhc_pre_fixture();
    let pre = reference_mhc_pre(fixture.input());
    let summary = deepseek_mhc_post(CudaDeepSeekMhcPostInput {
        tokens: fixture.tokens,
        hc_mult: fixture.hc_mult,
        hidden_size: fixture.hidden_size,
        x: &pre.layer_input,
        residual: &fixture.residual,
        post_layer_mix: &pre.post_mix,
        comb_res_mix: &pre.comb_mix,
    });
    if summary.status != SmokeStatus::Ok {
        return summary;
    }

    let expected = reference_mhc_post(CudaDeepSeekMhcPostInput {
        tokens: fixture.tokens,
        hc_mult: fixture.hc_mult,
        hidden_size: fixture.hidden_size,
        x: &pre.layer_input,
        residual: &fixture.residual,
        post_layer_mix: &pre.post_mix,
        comb_res_mix: &pre.comb_mix,
    });
    let matches_output = close_vec(&summary.output, &expected, 1e-5);
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
        "CUDA DeepSeek mHC post smoke mismatch: matches_output={} output_hash={} kernel_launches={}",
        matches_output, failed.output_hash, failed.kernel_launches
    ));
    failed
}

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
pub(crate) struct DeepSeekMhcPreFixture {
    pub(crate) tokens: u32,
    pub(crate) hc_mult: u32,
    pub(crate) hidden_size: u32,
    pub(crate) sinkhorn_repeat: u32,
    pub(crate) rms_eps: f32,
    pub(crate) hc_pre_eps: f32,
    pub(crate) hc_sinkhorn_eps: f32,
    pub(crate) hc_post_mult_value: f32,
    pub(crate) residual: Vec<f32>,
    pub(crate) fn_weights: Vec<f32>,
    pub(crate) hc_scale: Vec<f32>,
    pub(crate) hc_base: Vec<f32>,
}

impl DeepSeekMhcPreFixture {
    pub(crate) fn input(&self) -> CudaDeepSeekMhcPreInput<'_> {
        CudaDeepSeekMhcPreInput {
            tokens: self.tokens,
            hc_mult: self.hc_mult,
            hidden_size: self.hidden_size,
            sinkhorn_repeat: self.sinkhorn_repeat,
            rms_eps: self.rms_eps,
            hc_pre_eps: self.hc_pre_eps,
            hc_sinkhorn_eps: self.hc_sinkhorn_eps,
            hc_post_mult_value: self.hc_post_mult_value,
            residual: &self.residual,
            fn_weights: &self.fn_weights,
            hc_scale: &self.hc_scale,
            hc_base: &self.hc_base,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DeepSeekMhcPreReference {
    pub(crate) post_mix: Vec<f32>,
    pub(crate) comb_mix: Vec<f32>,
    pub(crate) layer_input: Vec<f32>,
}

pub(crate) fn deepseek_mhc_pre_fixture() -> DeepSeekMhcPreFixture {
    DeepSeekMhcPreFixture {
        tokens: 2,
        hc_mult: 2,
        hidden_size: 3,
        sinkhorn_repeat: 3,
        rms_eps: 1e-5,
        hc_pre_eps: 0.001,
        hc_sinkhorn_eps: 0.0001,
        hc_post_mult_value: 0.75,
        residual: vec![
            0.5, -1.0, 1.5, 0.25, 0.75, -0.5, -0.2, 0.4, 0.8, 1.2, -0.6, 0.3,
        ],
        fn_weights: (0..48)
            .map(|index| ((index % 11) as f32 - 5.0) * 0.07)
            .collect(),
        hc_scale: vec![1.1, 0.9, 1.25],
        hc_base: vec![0.05, -0.1, 0.12, -0.03, 0.2, -0.15, 0.08, 0.04],
    }
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

pub(crate) fn reference_mhc_pre(input: CudaDeepSeekMhcPreInput<'_>) -> DeepSeekMhcPreReference {
    let tokens = input.tokens as usize;
    let hc_mult = input.hc_mult as usize;
    let hidden_size = input.hidden_size as usize;
    let sinkhorn_repeat = input.sinkhorn_repeat as usize;
    let hc_mult2 = hc_mult * hc_mult;
    let hc_mult3 = hc_mult * 2 + hc_mult2;
    let hc_hidden_size = hc_mult * hidden_size;
    let mut post_mix = vec![0.0f32; tokens * hc_mult];
    let mut comb_mix = vec![0.0f32; tokens * hc_mult2];
    let mut layer_input = vec![0.0f32; tokens * hidden_size];
    let mut mixes = vec![0.0f32; hc_mult3];
    for token in 0..tokens {
        let token_start = token * hc_hidden_size;
        let residual_token = &input.residual[token_start..token_start + hc_hidden_size];
        let sqrsum = residual_token
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let rms_scale = 1.0 / (sqrsum / hc_hidden_size as f32 + input.rms_eps).sqrt();
        for (mix, mix_out) in mixes.iter_mut().enumerate() {
            let row_start = mix * hc_hidden_size;
            let row = &input.fn_weights[row_start..row_start + hc_hidden_size];
            *mix_out = row
                .iter()
                .zip(residual_token.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
                * rms_scale;
        }

        let mut pre_mix = vec![0.0f32; hc_mult];
        for channel in 0..hc_mult {
            pre_mix[channel] = sigmoid(mixes[channel] * input.hc_scale[0] + input.hc_base[channel])
                + input.hc_pre_eps;
            post_mix[token * hc_mult + channel] = sigmoid(
                mixes[hc_mult + channel] * input.hc_scale[1] + input.hc_base[hc_mult + channel],
            ) * input.hc_post_mult_value;
        }

        let comb_offset = token * hc_mult2;
        for row in 0..hc_mult {
            let logits_start = 2 * hc_mult + row * hc_mult;
            let logits = (0..hc_mult)
                .map(|col| {
                    mixes[logits_start + col] * input.hc_scale[2]
                        + input.hc_base[logits_start + col]
                })
                .collect::<Vec<_>>();
            let row_softmax = softmax(&logits);
            for col in 0..hc_mult {
                comb_mix[comb_offset + row * hc_mult + col] =
                    row_softmax[col] + input.hc_sinkhorn_eps;
            }
        }
        normalize_comb_columns(
            &mut comb_mix[comb_offset..comb_offset + hc_mult2],
            hc_mult,
            input.hc_sinkhorn_eps,
        );
        for _ in 0..sinkhorn_repeat.saturating_sub(1) {
            normalize_comb_rows(
                &mut comb_mix[comb_offset..comb_offset + hc_mult2],
                hc_mult,
                input.hc_sinkhorn_eps,
            );
            normalize_comb_columns(
                &mut comb_mix[comb_offset..comb_offset + hc_mult2],
                hc_mult,
                input.hc_sinkhorn_eps,
            );
        }

        for hidden in 0..hidden_size {
            let mut value = 0.0f32;
            for channel in 0..hc_mult {
                value += pre_mix[channel] * residual_token[channel * hidden_size + hidden];
            }
            layer_input[token * hidden_size + hidden] = value;
        }
    }
    DeepSeekMhcPreReference {
        post_mix,
        comb_mix,
        layer_input,
    }
}

pub(crate) fn reference_mhc_post(input: CudaDeepSeekMhcPostInput<'_>) -> Vec<f32> {
    let tokens = input.tokens as usize;
    let hc_mult = input.hc_mult as usize;
    let hidden_size = input.hidden_size as usize;
    let hc_hidden_size = hc_mult * hidden_size;
    let mut output = vec![0.0f32; tokens * hc_hidden_size];
    for token in 0..tokens {
        let residual_offset = token * hc_hidden_size;
        let comb_offset = token * hc_mult * hc_mult;
        for out_channel in 0..hc_mult {
            for hidden in 0..hidden_size {
                let mut mixed = 0.0f32;
                for in_channel in 0..hc_mult {
                    mixed += input.comb_res_mix[comb_offset + in_channel * hc_mult + out_channel]
                        * input.residual[residual_offset + in_channel * hidden_size + hidden];
                }
                mixed += input.post_layer_mix[token * hc_mult + out_channel]
                    * input.x[token * hidden_size + hidden];
                output[residual_offset + out_channel * hidden_size + hidden] = mixed;
            }
        }
    }
    output
}

fn close_vec(actual: &[f32], expected: &[f32], tolerance: f32) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| (actual - expected).abs() <= tolerance)
}

fn normalize_comb_rows(comb: &mut [f32], hc_mult: usize, eps: f32) {
    for row in 0..hc_mult {
        let row_start = row * hc_mult;
        let sum = comb[row_start..row_start + hc_mult].iter().sum::<f32>();
        for col in 0..hc_mult {
            comb[row_start + col] /= sum + eps;
        }
    }
}

fn normalize_comb_columns(comb: &mut [f32], hc_mult: usize, eps: f32) {
    for col in 0..hc_mult {
        let mut sum = 0.0f32;
        for row in 0..hc_mult {
            sum += comb[row * hc_mult + col];
        }
        for row in 0..hc_mult {
            comb[row * hc_mult + col] /= sum + eps;
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = logits
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let sum = values.iter().sum::<f32>();
    if sum > 0.0 {
        for value in values.iter_mut() {
            *value /= sum;
        }
    }
    values
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
