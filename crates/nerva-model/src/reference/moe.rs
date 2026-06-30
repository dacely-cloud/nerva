use nerva_core::types::error::{NervaError, Result};

use crate::common::math::silu;
use crate::common::validate::require_len;
use crate::reference::router::DeepSeekRouteSelection;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekRoutedMoeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub top_k: usize,
    pub swiglu_limit: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekFullRoutedMoeConfig {
    pub hidden_size: usize,
    pub routed_intermediate_size: usize,
    pub top_k: usize,
    pub swiglu_limit: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekRoutedMoeWeights<'a> {
    pub w_gate: &'a [f32],
    pub w_up: &'a [f32],
    pub w_down: &'a [f32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekSharedMoeWeights<'a> {
    pub w_gate: &'a [f32],
    pub w_up: &'a [f32],
    pub w_down: &'a [f32],
    pub intermediate_size: usize,
}

pub fn deepseek_full_routed_moe_forward(
    input: &[f32],
    route: &DeepSeekRouteSelection,
    routed_weights: DeepSeekRoutedMoeWeights<'_>,
    shared_weights: Option<DeepSeekSharedMoeWeights<'_>>,
    config: DeepSeekFullRoutedMoeConfig,
    output: &mut [f32],
) -> Result<()> {
    validate_swiglu_limit(config.swiglu_limit)?;
    deepseek_routed_moe_forward(
        input,
        &route.expert_ids,
        &route.weights,
        routed_weights.w_gate,
        routed_weights.w_up,
        routed_weights.w_down,
        DeepSeekRoutedMoeConfig {
            hidden_size: config.hidden_size,
            intermediate_size: config.routed_intermediate_size,
            top_k: config.top_k,
            swiglu_limit: config.swiglu_limit,
        },
        output,
    )?;

    if let Some(shared) = shared_weights {
        add_deepseek_shared_moe_forward(
            input,
            shared,
            config.hidden_size,
            config.swiglu_limit,
            output,
        )?;
    }

    Ok(())
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

fn add_deepseek_shared_moe_forward(
    input: &[f32],
    weights: DeepSeekSharedMoeWeights<'_>,
    hidden_size: usize,
    swiglu_limit: Option<f32>,
    output: &mut [f32],
) -> Result<()> {
    validate_deepseek_shared_moe(input, weights, hidden_size, swiglu_limit, output)?;

    let mut activation = vec![0.0f32; weights.intermediate_size];
    for (row, activation) in activation.iter_mut().enumerate() {
        let start = row * hidden_size;
        let end = start + hidden_size;
        let gate = dot(&weights.w_gate[start..end], input);
        let up = dot(&weights.w_up[start..end], input);
        *activation = deepseek_swiglu(gate, up, swiglu_limit);
    }

    for (hidden, output) in output.iter_mut().enumerate() {
        let start = hidden * weights.intermediate_size;
        let end = start + weights.intermediate_size;
        *output += dot(&weights.w_down[start..end], &activation);
    }

    Ok(())
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
    validate_swiglu_limit(config.swiglu_limit)?;
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

fn validate_deepseek_shared_moe(
    input: &[f32],
    weights: DeepSeekSharedMoeWeights<'_>,
    hidden_size: usize,
    swiglu_limit: Option<f32>,
    output: &[f32],
) -> Result<()> {
    if hidden_size == 0 || weights.intermediate_size == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek shared MoE dimensions must be non-zero".to_string(),
        });
    }
    validate_swiglu_limit(swiglu_limit)?;
    require_len("DeepSeek shared MoE input", input.len(), hidden_size)?;
    require_len("DeepSeek shared MoE output", output.len(), hidden_size)?;
    require_len(
        "DeepSeek shared MoE gate weights",
        weights.w_gate.len(),
        weights.intermediate_size * hidden_size,
    )?;
    require_len(
        "DeepSeek shared MoE up weights",
        weights.w_up.len(),
        weights.intermediate_size * hidden_size,
    )?;
    require_len(
        "DeepSeek shared MoE down weights",
        weights.w_down.len(),
        hidden_size * weights.intermediate_size,
    )?;
    if input.iter().any(|value| !value.is_finite())
        || weights.w_gate.iter().any(|value| !value.is_finite())
        || weights.w_up.iter().any(|value| !value.is_finite())
        || weights.w_down.iter().any(|value| !value.is_finite())
    {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek shared MoE tensors must be finite".to_string(),
        });
    }
    Ok(())
}

fn validate_swiglu_limit(swiglu_limit: Option<f32>) -> Result<()> {
    if let Some(limit) = swiglu_limit
        && (!limit.is_finite() || limit <= 0.0)
    {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek routed MoE SwiGLU limit must be finite and positive".to_string(),
        });
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}
