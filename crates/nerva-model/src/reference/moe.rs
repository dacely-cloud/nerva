use nerva_core::types::error::{NervaError, Result};

use crate::common::math::silu;
use crate::common::validate::require_len;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekRoutedMoeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub top_k: usize,
    pub swiglu_limit: Option<f32>,
}

pub fn deepseek_routed_moe_forward(
    input: &[f32],
    expert_ids: &[usize],
    expert_weights: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    config: DeepSeekRoutedMoeConfig,
    output: &mut [f32],
) -> Result<()> {
    validate_deepseek_routed_moe(
        input,
        expert_ids,
        expert_weights,
        w_gate,
        w_up,
        w_down,
        config,
        output,
    )?;

    output.fill(0.0);
    let expert_stride = config.intermediate_size * config.hidden_size;
    let down_expert_stride = config.hidden_size * config.intermediate_size;
    let num_experts = w_gate.len() / expert_stride;
    let mut activation = vec![0.0f32; config.intermediate_size];

    for (&expert, &route_weight) in expert_ids.iter().zip(expert_weights.iter()) {
        if expert >= num_experts {
            return Err(NervaError::InvalidArgument {
                reason: "DeepSeek routed MoE expert id is out of range".to_string(),
            });
        }
        let gate_base = expert * expert_stride;
        let down_base = expert * down_expert_stride;

        for row in 0..config.intermediate_size {
            let start = gate_base + row * config.hidden_size;
            let end = start + config.hidden_size;
            let gate = dot(&w_gate[start..end], input);
            let up = dot(&w_up[start..end], input);
            activation[row] = deepseek_swiglu(gate, up, config.swiglu_limit);
        }

        for hidden in 0..config.hidden_size {
            let start = down_base + hidden * config.intermediate_size;
            let end = start + config.intermediate_size;
            output[hidden] += route_weight * dot(&w_down[start..end], &activation);
        }
    }

    Ok(())
}

pub fn deepseek_swiglu(gate: f32, up: f32, swiglu_limit: Option<f32>) -> f32 {
    match swiglu_limit {
        Some(limit) => {
            let clamped_gate = gate.min(limit);
            let clamped_up = up.clamp(-limit, limit);
            silu(clamped_gate) * clamped_up
        }
        None => silu(gate) * up,
    }
}

fn validate_deepseek_routed_moe(
    input: &[f32],
    expert_ids: &[usize],
    expert_weights: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    config: DeepSeekRoutedMoeConfig,
    output: &[f32],
) -> Result<()> {
    if config.hidden_size == 0 || config.intermediate_size == 0 || config.top_k == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek routed MoE dimensions must be non-zero".to_string(),
        });
    }
    require_len("DeepSeek routed MoE input", input.len(), config.hidden_size)?;
    require_len(
        "DeepSeek routed MoE output",
        output.len(),
        config.hidden_size,
    )?;
    require_len(
        "DeepSeek routed MoE expert ids",
        expert_ids.len(),
        config.top_k,
    )?;
    require_len(
        "DeepSeek routed MoE expert weights",
        expert_weights.len(),
        config.top_k,
    )?;
    if let Some(limit) = config.swiglu_limit
        && (!limit.is_finite() || limit <= 0.0)
    {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek routed MoE SwiGLU limit must be finite and positive".to_string(),
        });
    }
    if input.iter().any(|value| !value.is_finite())
        || expert_weights.iter().any(|value| !value.is_finite())
        || w_gate.iter().any(|value| !value.is_finite())
        || w_up.iter().any(|value| !value.is_finite())
        || w_down.iter().any(|value| !value.is_finite())
    {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek routed MoE tensors must be finite".to_string(),
        });
    }

    let expert_stride = config.intermediate_size * config.hidden_size;
    let down_expert_stride = config.hidden_size * config.intermediate_size;
    if w_gate.is_empty()
        || w_gate.len() != w_up.len()
        || !w_gate.len().is_multiple_of(expert_stride)
    {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek routed MoE gate/up weights have invalid shape".to_string(),
        });
    }
    let num_experts = w_gate.len() / expert_stride;
    require_len(
        "DeepSeek routed MoE down weights",
        w_down.len(),
        num_experts * down_expert_stride,
    )?;
    for expert in expert_ids {
        if *expert >= num_experts {
            return Err(NervaError::InvalidArgument {
                reason: "DeepSeek routed MoE expert id is out of range".to_string(),
            });
        }
    }

    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}
