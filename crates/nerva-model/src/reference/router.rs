use nerva_core::types::error::{NervaError, Result};

use crate::common::math::sigmoid;
use crate::common::validate::require_len;

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekRouteSelection {
    pub expert_ids: Vec<usize>,
    pub weights: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekRouterScoring {
    Sigmoid,
    Softmax,
    SqrtSoftplus,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekV3GroupedRouterConfig {
    pub top_k: usize,
    pub num_expert_groups: usize,
    pub top_k_groups: usize,
    pub scoring: DeepSeekRouterScoring,
    pub renormalize: bool,
    pub routed_scaling_factor: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekV4RouterConfig {
    pub top_k: usize,
    pub renormalize: bool,
    pub routed_scaling_factor: f32,
}

pub fn deepseek_v3_grouped_route(
    logits: &[f32],
    correction_bias: Option<&[f32]>,
    config: DeepSeekV3GroupedRouterConfig,
) -> Result<DeepSeekRouteSelection> {
    validate_route_common(logits, config.top_k, config.routed_scaling_factor)?;
    if config.num_expert_groups == 0 || config.top_k_groups == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek V3 grouped routing requires non-zero group counts".to_string(),
        });
    }
    if config.top_k_groups > config.num_expert_groups {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek V3 top_k_groups cannot exceed num_expert_groups".to_string(),
        });
    }
    if logits.len() <= config.num_expert_groups
        || !logits.len().is_multiple_of(config.num_expert_groups)
    {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek V3 expert count must be divisible by num_expert_groups".to_string(),
        });
    }
    if let Some(bias) = correction_bias {
        require_len(
            "DeepSeek V3 router correction bias",
            bias.len(),
            logits.len(),
        )?;
    }

    let scores = score_logits(logits, config.scoring)?;
    let scores_for_choice = biased_scores(&scores, correction_bias);
    let experts_per_group = logits.len() / config.num_expert_groups;
    let mut group_scores = vec![0.0f32; config.num_expert_groups];
    for (group, group_score) in group_scores.iter_mut().enumerate() {
        let start = group * experts_per_group;
        let end = start + experts_per_group;
        *group_score = if correction_bias.is_some() {
            top_n_sum(&scores_for_choice[start..end], 2)
        } else {
            scores_for_choice[start..end]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
        };
    }

    let selected_groups = top_k_indices(&group_scores, config.top_k_groups);
    let mut choice_scores = vec![f32::NEG_INFINITY; logits.len()];
    for group in selected_groups {
        let start = group * experts_per_group;
        let end = start + experts_per_group;
        choice_scores[start..end].copy_from_slice(&scores_for_choice[start..end]);
    }

    let expert_ids = top_k_indices(&choice_scores, config.top_k);
    let mut weights = expert_ids
        .iter()
        .map(|expert| scores[*expert])
        .collect::<Vec<_>>();
    finish_route_weights(
        &mut weights,
        config.renormalize,
        config.routed_scaling_factor,
    )?;
    Ok(DeepSeekRouteSelection {
        expert_ids,
        weights,
    })
}

pub fn deepseek_v4_sqrtsoftplus_route(
    logits: &[f32],
    correction_bias: Option<&[f32]>,
    hash_expert_ids: Option<&[usize]>,
    config: DeepSeekV4RouterConfig,
) -> Result<DeepSeekRouteSelection> {
    validate_route_common(logits, config.top_k, config.routed_scaling_factor)?;
    if let Some(bias) = correction_bias {
        require_len(
            "DeepSeek V4 router correction bias",
            bias.len(),
            logits.len(),
        )?;
    }
    if let Some(ids) = hash_expert_ids {
        require_len("DeepSeek V4 hash route table row", ids.len(), config.top_k)?;
        for id in ids {
            if *id >= logits.len() {
                return Err(NervaError::InvalidArgument {
                    reason: "DeepSeek V4 hash route expert id is out of range".to_string(),
                });
            }
        }
    }

    let scores = score_logits(logits, DeepSeekRouterScoring::SqrtSoftplus)?;
    let expert_ids = if let Some(ids) = hash_expert_ids {
        ids.to_vec()
    } else {
        top_k_indices(&biased_scores(&scores, correction_bias), config.top_k)
    };
    let mut weights = expert_ids
        .iter()
        .map(|expert| scores[*expert])
        .collect::<Vec<_>>();
    finish_route_weights(
        &mut weights,
        config.renormalize,
        config.routed_scaling_factor,
    )?;
    Ok(DeepSeekRouteSelection {
        expert_ids,
        weights,
    })
}

fn validate_route_common(logits: &[f32], top_k: usize, scaling_factor: f32) -> Result<()> {
    if logits.is_empty() {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek routing requires at least one expert".to_string(),
        });
    }
    if top_k == 0 || top_k > logits.len() {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek routing top_k must be in 1..=num_experts".to_string(),
        });
    }
    if !scaling_factor.is_finite() || scaling_factor <= 0.0 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek routed scaling factor must be finite and positive".to_string(),
        });
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek router logits must be finite".to_string(),
        });
    }
    Ok(())
}

fn score_logits(logits: &[f32], scoring: DeepSeekRouterScoring) -> Result<Vec<f32>> {
    match scoring {
        DeepSeekRouterScoring::Sigmoid => Ok(logits.iter().map(|value| sigmoid(*value)).collect()),
        DeepSeekRouterScoring::Softmax => Ok(softmax(logits)),
        DeepSeekRouterScoring::SqrtSoftplus => {
            Ok(logits.iter().map(|value| softplus(*value).sqrt()).collect())
        }
    }
}

fn biased_scores(scores: &[f32], correction_bias: Option<&[f32]>) -> Vec<f32> {
    correction_bias.map_or_else(
        || scores.to_vec(),
        |bias| {
            scores
                .iter()
                .zip(bias.iter())
                .map(|(score, bias)| score + bias)
                .collect()
        },
    )
}

fn finish_route_weights(
    weights: &mut [f32],
    renormalize: bool,
    routed_scaling_factor: f32,
) -> Result<()> {
    if renormalize {
        let sum = weights.iter().sum::<f32>();
        if sum <= 0.0 || !sum.is_finite() {
            return Err(NervaError::InvalidArgument {
                reason: "DeepSeek routing produced invalid top-k weight sum".to_string(),
            });
        }
        for weight in weights.iter_mut() {
            *weight /= sum;
        }
    }
    if routed_scaling_factor != 1.0 {
        for weight in weights.iter_mut() {
            *weight *= routed_scaling_factor;
        }
    }
    Ok(())
}

fn top_n_sum(values: &[f32], count: usize) -> f32 {
    let mut top = vec![f32::NEG_INFINITY; count.min(values.len())];
    for value in values.iter().copied() {
        for slot in 0..top.len() {
            if value > top[slot] {
                for shift in (slot + 1..top.len()).rev() {
                    top[shift] = top[shift - 1];
                }
                top[slot] = value;
                break;
            }
        }
    }
    top.into_iter().filter(|value| value.is_finite()).sum()
}

fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    indices.truncate(k);
    indices
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for logit in logits {
        let value = (*logit - max).exp();
        sum += value;
        values.push(value);
    }
    if sum > 0.0 {
        for value in &mut values {
            *value /= sum;
        }
    }
    values
}
