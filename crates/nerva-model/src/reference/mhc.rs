use nerva_core::types::error::{NervaError, Result};

use crate::common::math::sigmoid;
use crate::common::validate::require_len;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekMhcConfig {
    pub hc_mult: usize,
    pub hidden_size: usize,
    pub rms_eps: f32,
    pub hc_pre_eps: f32,
    pub hc_sinkhorn_eps: f32,
    pub hc_post_mult_value: f32,
    pub sinkhorn_repeat: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekMhcPreOutput {
    pub post_mix: Vec<f32>,
    pub comb_mix: Vec<f32>,
    pub layer_input: Vec<f32>,
}

pub fn deepseek_mhc_pre_torch_reference(
    residual: &[f32],
    fn_weights: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    config: DeepSeekMhcConfig,
) -> Result<DeepSeekMhcPreOutput> {
    validate_mhc_config(config)?;
    require_len("DeepSeek V4 mHC scale", hc_scale.len(), 3)?;
    let hc_mult2 = config.hc_mult * config.hc_mult;
    let hc_mult3 = config.hc_mult * 2 + hc_mult2;
    let hc_hidden_size = config.hc_mult * config.hidden_size;
    require_len("DeepSeek V4 mHC base", hc_base.len(), hc_mult3)?;
    require_len(
        "DeepSeek V4 mHC fn",
        fn_weights.len(),
        hc_mult3 * hc_hidden_size,
    )?;
    if residual.is_empty() || !residual.len().is_multiple_of(hc_hidden_size) {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek V4 mHC residual must have shape tokens x hc_mult x hidden_size"
                .to_string(),
        });
    }
    validate_finite("DeepSeek V4 mHC residual", residual)?;
    validate_finite("DeepSeek V4 mHC fn", fn_weights)?;
    validate_finite("DeepSeek V4 mHC scale", hc_scale)?;
    validate_finite("DeepSeek V4 mHC base", hc_base)?;

    let tokens = residual.len() / hc_hidden_size;
    let mut post_mix = vec![0.0f32; tokens * config.hc_mult];
    let mut comb_mix = vec![0.0f32; tokens * hc_mult2];
    let mut layer_input = vec![0.0f32; tokens * config.hidden_size];
    let mut mixes = vec![0.0f32; hc_mult3];

    for token in 0..tokens {
        let residual_offset = token * hc_hidden_size;
        let residual_token = &residual[residual_offset..residual_offset + hc_hidden_size];
        let sqrsum = residual_token
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let rms_scale = (sqrsum / hc_hidden_size as f32 + config.rms_eps)
            .sqrt()
            .recip();

        for mix in 0..hc_mult3 {
            let row = &fn_weights[mix * hc_hidden_size..(mix + 1) * hc_hidden_size];
            mixes[mix] = row
                .iter()
                .zip(residual_token.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
                * rms_scale;
        }

        let mut pre_mix = vec![0.0f32; config.hc_mult];
        for index in 0..config.hc_mult {
            pre_mix[index] =
                sigmoid(mixes[index] * hc_scale[0] + hc_base[index]) + config.hc_pre_eps;
            post_mix[token * config.hc_mult + index] = sigmoid(
                mixes[config.hc_mult + index] * hc_scale[1] + hc_base[config.hc_mult + index],
            ) * config.hc_post_mult_value;
        }

        let comb_offset = token * hc_mult2;
        for row in 0..config.hc_mult {
            let row_start = comb_offset + row * config.hc_mult;
            let logits_start = 2 * config.hc_mult + row * config.hc_mult;
            let logits = (0..config.hc_mult)
                .map(|col| mixes[logits_start + col] * hc_scale[2] + hc_base[logits_start + col])
                .collect::<Vec<_>>();
            let row_softmax = softmax(&logits);
            for col in 0..config.hc_mult {
                comb_mix[row_start + col] = row_softmax[col] + config.hc_sinkhorn_eps;
            }
        }
        normalize_comb_columns(&mut comb_mix[comb_offset..comb_offset + hc_mult2], config);
        for _ in 0..config.sinkhorn_repeat.saturating_sub(1) {
            normalize_comb_rows(&mut comb_mix[comb_offset..comb_offset + hc_mult2], config);
            normalize_comb_columns(&mut comb_mix[comb_offset..comb_offset + hc_mult2], config);
        }

        for hidden in 0..config.hidden_size {
            let mut value = 0.0f32;
            for channel in 0..config.hc_mult {
                value += pre_mix[channel] * residual_token[channel * config.hidden_size + hidden];
            }
            layer_input[token * config.hidden_size + hidden] = value;
        }
    }

    Ok(DeepSeekMhcPreOutput {
        post_mix,
        comb_mix,
        layer_input,
    })
}

pub fn deepseek_mhc_post_torch_reference(
    x: &[f32],
    residual: &[f32],
    post_layer_mix: &[f32],
    comb_res_mix: &[f32],
    hc_mult: usize,
    hidden_size: usize,
) -> Result<Vec<f32>> {
    if hc_mult == 0 || hidden_size == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek V4 mHC post requires non-zero hc_mult and hidden_size".to_string(),
        });
    }
    let hc_hidden_size = hc_mult * hidden_size;
    if residual.is_empty() || !residual.len().is_multiple_of(hc_hidden_size) {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek V4 mHC post residual must have shape tokens x hc_mult x hidden_size"
                .to_string(),
        });
    }
    let tokens = residual.len() / hc_hidden_size;
    require_len("DeepSeek V4 mHC post x", x.len(), tokens * hidden_size)?;
    require_len(
        "DeepSeek V4 mHC post layer mix",
        post_layer_mix.len(),
        tokens * hc_mult,
    )?;
    require_len(
        "DeepSeek V4 mHC comb residual mix",
        comb_res_mix.len(),
        tokens * hc_mult * hc_mult,
    )?;
    validate_finite("DeepSeek V4 mHC post x", x)?;
    validate_finite("DeepSeek V4 mHC post residual", residual)?;
    validate_finite("DeepSeek V4 mHC post layer mix", post_layer_mix)?;
    validate_finite("DeepSeek V4 mHC comb residual mix", comb_res_mix)?;

    let mut output = vec![0.0f32; residual.len()];
    for token in 0..tokens {
        let residual_offset = token * hc_hidden_size;
        let comb_offset = token * hc_mult * hc_mult;
        for out_channel in 0..hc_mult {
            for hidden in 0..hidden_size {
                let mut mixed = 0.0f32;
                for in_channel in 0..hc_mult {
                    mixed += comb_res_mix[comb_offset + in_channel * hc_mult + out_channel]
                        * residual[residual_offset + in_channel * hidden_size + hidden];
                }
                mixed +=
                    post_layer_mix[token * hc_mult + out_channel] * x[token * hidden_size + hidden];
                output[residual_offset + out_channel * hidden_size + hidden] = mixed;
            }
        }
    }
    Ok(output)
}

pub fn deepseek_hc_head_torch_reference(
    hidden_states: &[f32],
    fn_weights: &[f32],
    hc_scale: f32,
    hc_base: &[f32],
    hc_mult: usize,
    hidden_size: usize,
    rms_eps: f32,
    hc_eps: f32,
) -> Result<Vec<f32>> {
    if hc_mult == 0 || hidden_size == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek V4 HC head requires non-zero hc_mult and hidden_size".to_string(),
        });
    }
    let hc_hidden_size = hc_mult * hidden_size;
    if hidden_states.is_empty() || !hidden_states.len().is_multiple_of(hc_hidden_size) {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek V4 HC head input must have shape tokens x hc_mult x hidden_size"
                .to_string(),
        });
    }
    let tokens = hidden_states.len() / hc_hidden_size;
    require_len(
        "DeepSeek V4 HC head fn",
        fn_weights.len(),
        hc_mult * hc_hidden_size,
    )?;
    require_len("DeepSeek V4 HC head base", hc_base.len(), hc_mult)?;
    if !hc_scale.is_finite() || !rms_eps.is_finite() || !hc_eps.is_finite() {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek V4 HC head scalar parameters must be finite".to_string(),
        });
    }
    validate_finite("DeepSeek V4 HC head hidden states", hidden_states)?;
    validate_finite("DeepSeek V4 HC head fn", fn_weights)?;
    validate_finite("DeepSeek V4 HC head base", hc_base)?;

    let mut output = vec![0.0f32; tokens * hidden_size];
    for token in 0..tokens {
        let token_offset = token * hc_hidden_size;
        let token_values = &hidden_states[token_offset..token_offset + hc_hidden_size];
        let sqrsum = token_values.iter().map(|value| value * value).sum::<f32>();
        let rms_scale = (sqrsum / hc_hidden_size as f32 + rms_eps).sqrt().recip();
        let mut gates = vec![0.0f32; hc_mult];
        for channel in 0..hc_mult {
            let row = &fn_weights[channel * hc_hidden_size..(channel + 1) * hc_hidden_size];
            let mix = row
                .iter()
                .zip(token_values.iter())
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
                * rms_scale;
            gates[channel] = sigmoid(mix * hc_scale + hc_base[channel]) + hc_eps;
        }
        for hidden in 0..hidden_size {
            let mut value = 0.0f32;
            for channel in 0..hc_mult {
                value += gates[channel] * token_values[channel * hidden_size + hidden];
            }
            output[token * hidden_size + hidden] = value;
        }
    }
    Ok(output)
}

fn validate_mhc_config(config: DeepSeekMhcConfig) -> Result<()> {
    if config.hc_mult == 0 || config.hidden_size == 0 || config.sinkhorn_repeat == 0 {
        return Err(NervaError::InvalidArgument {
            reason:
                "DeepSeek V4 mHC config requires non-zero hc_mult, hidden_size, and sinkhorn_repeat"
                    .to_string(),
        });
    }
    for (name, value) in [
        ("rms_eps", config.rms_eps),
        ("hc_pre_eps", config.hc_pre_eps),
        ("hc_sinkhorn_eps", config.hc_sinkhorn_eps),
        ("hc_post_mult_value", config.hc_post_mult_value),
    ] {
        if !value.is_finite() {
            return Err(NervaError::InvalidArgument {
                reason: format!("DeepSeek V4 mHC {name} must be finite"),
            });
        }
    }
    Ok(())
}

fn validate_finite(label: &str, values: &[f32]) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(NervaError::InvalidArgument {
            reason: format!("{label} values must be finite"),
        });
    }
    Ok(())
}

fn normalize_comb_rows(comb: &mut [f32], config: DeepSeekMhcConfig) {
    for row in 0..config.hc_mult {
        let row_start = row * config.hc_mult;
        let sum = comb[row_start..row_start + config.hc_mult]
            .iter()
            .sum::<f32>();
        for col in 0..config.hc_mult {
            comb[row_start + col] /= sum + config.hc_sinkhorn_eps;
        }
    }
}

fn normalize_comb_columns(comb: &mut [f32], config: DeepSeekMhcConfig) {
    for col in 0..config.hc_mult {
        let mut sum = 0.0f32;
        for row in 0..config.hc_mult {
            sum += comb[row * config.hc_mult + col];
        }
        for row in 0..config.hc_mult {
            comb[row * config.hc_mult + col] /= sum + config.hc_sinkhorn_eps;
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
