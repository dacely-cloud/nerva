use nerva_core::types::error::{NervaError, Result};
use nerva_cuda::decode::hf_sequence::weight_plan::{
    CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT, CUDA_HF_WEIGHT_STRATEGY_GPU_STAGED,
    CudaHfDecodeSequenceWeightBlock,
};
use nerva_model::causal_lm::types::{HfCausalLmLayer, HfCausalLmLoaded};
use nerva_model::precision::block::gdn::PrecisionGatedDeltaNetMoeEncodedView;
use nerva_model::precision::block::model::PrecisionTransformerBlockEncodedView;
use nerva_model::precision::block::moe::PrecisionMoeTransformerBlockEncodedView;
use nerva_model::weights::layout::entry::WeightBlockRole;
use nerva_model::weights::manifest::HfTensorManifestEntry;

use crate::weights::execution::plan::ResidentWeightExecutionPlan;
use crate::weights::execution::strategy::ResidentWeightExecutionStrategy;

pub(super) fn cuda_weight_descriptors(
    loaded: &HfCausalLmLoaded,
    plan: &ResidentWeightExecutionPlan,
) -> Result<Vec<CudaHfDecodeSequenceWeightBlock>> {
    let manifest = &loaded.summary.manifest;
    if plan.steps.len() != manifest.entries.len() {
        return Err(NervaError::InvalidArgument {
            reason: "CUDA HF weight descriptor plan length does not match manifest".to_string(),
        });
    }
    let mut offset_bytes = 0u64;
    let mut descriptors = Vec::with_capacity(plan.steps.len());
    for (step, entry) in plan.steps.iter().zip(&manifest.entries) {
        if step.name != entry.name || step.bytes != entry.bytes {
            return Err(NervaError::InvalidArgument {
                reason: "CUDA HF weight descriptor step does not match manifest".to_string(),
            });
        }
        let source = weight_source(loaded, entry)?;
        let bytes = step.bytes as u64;
        descriptors.push(CudaHfDecodeSequenceWeightBlock {
            host_source: source.as_ptr(),
            block_id: step.block_id.0,
            block_version: step.block_version,
            offset_bytes,
            bytes,
            strategy: cuda_weight_strategy(step.strategy)?,
            reserved: 0,
            ..CudaHfDecodeSequenceWeightBlock::default()
        });
        offset_bytes =
            offset_bytes
                .checked_add(bytes)
                .ok_or_else(|| NervaError::AllocationFailed {
                    bytes: step.bytes,
                    reason: "CUDA HF weight descriptor offset overflow".to_string(),
                })?;
    }
    Ok(descriptors)
}

fn weight_source<'a>(
    loaded: &'a HfCausalLmLoaded,
    entry: &HfTensorManifestEntry,
) -> Result<&'a [u16]> {
    let source = match entry.role {
        WeightBlockRole::TokenEmbedding => loaded.model.token_embeddings(),
        WeightBlockRole::FinalNorm => loaded.model.final_norm_weight(),
        WeightBlockRole::LmHead => loaded.model.lm_head(),
        role => layer_weight_source(loaded, entry, role)?,
    };
    if source.len() * 2 != entry.bytes {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "CUDA HF descriptor source {} has wrong byte length",
                entry.name
            ),
        });
    }
    Ok(source)
}

fn layer_weight_source<'a>(
    loaded: &'a HfCausalLmLoaded,
    entry: &HfTensorManifestEntry,
    role: WeightBlockRole,
) -> Result<&'a [u16]> {
    let layer_index = entry.layer.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("CUDA HF descriptor source {} has no layer", entry.name),
    })? as usize;
    let layer =
        loaded
            .model
            .causal_layer(layer_index)
            .ok_or_else(|| NervaError::InvalidArgument {
                reason: format!("CUDA HF descriptor layer {layer_index} is unavailable"),
            })?;
    match layer {
        HfCausalLmLayer::Dense(layer) => {
            dense_layer_weight_source(layer.encoded_view(), entry, role)
        }
        HfCausalLmLayer::SparseMoe(layer) => {
            sparse_moe_layer_weight_source(layer.encoded_view(), entry, role)
        }
        HfCausalLmLayer::GatedDeltaNetMoe(layer) => {
            gdn_moe_layer_weight_source(layer.encoded_view(), entry, role)
        }
    }
}

fn gdn_moe_layer_weight_source<'a>(
    view: PrecisionGatedDeltaNetMoeEncodedView<'a>,
    entry: &HfTensorManifestEntry,
    role: WeightBlockRole,
) -> Result<&'a [u16]> {
    match role {
        WeightBlockRole::AttentionNorm => Ok(view.rms_attn_weight),
        WeightBlockRole::LinearConvProjection => Ok(view.linear_conv),
        WeightBlockRole::LinearQkvProjection => Ok(view.linear_qkv),
        WeightBlockRole::LinearZProjection => Ok(view.linear_z),
        WeightBlockRole::LinearBProjection => Ok(view.linear_b),
        WeightBlockRole::LinearAProjection => Ok(view.linear_a),
        WeightBlockRole::LinearDtBias => Ok(view.linear_dt_bias),
        WeightBlockRole::LinearALog => Ok(view.linear_a_log_bits),
        WeightBlockRole::LinearNorm => Ok(view.linear_norm_bits),
        WeightBlockRole::LinearOutputProjection => Ok(view.linear_out),
        WeightBlockRole::MlpNorm => Ok(view.rms_mlp_weight),
        WeightBlockRole::RouterProjection => Ok(view.router),
        WeightBlockRole::ExpertGateProjection => gdn_moe_expert_gate_up_slice(view, entry, false),
        WeightBlockRole::ExpertUpProjection => gdn_moe_expert_gate_up_slice(view, entry, true),
        WeightBlockRole::ExpertGateUpProjection => Ok(view.expert_gate_up),
        WeightBlockRole::ExpertDownProjection => gdn_moe_expert_down_slice(view, entry),
        WeightBlockRole::SharedExpertGateProjection => Ok(view.shared_expert_gate),
        WeightBlockRole::SharedExpertUpProjection => Ok(view.shared_expert_up),
        WeightBlockRole::SharedExpertDownProjection => Ok(view.shared_expert_down),
        WeightBlockRole::SharedExpertRouterProjection => Ok(view.shared_expert_router),
        WeightBlockRole::QueryProjection
        | WeightBlockRole::QueryNorm
        | WeightBlockRole::QueryBias
        | WeightBlockRole::KeyProjection
        | WeightBlockRole::KeyNorm
        | WeightBlockRole::KeyBias
        | WeightBlockRole::ValueProjection
        | WeightBlockRole::ValueBias
        | WeightBlockRole::OutputProjection
        | WeightBlockRole::OutputBias
        | WeightBlockRole::GateProjection
        | WeightBlockRole::UpProjection
        | WeightBlockRole::DownProjection => Err(NervaError::InvalidArgument {
            reason: format!(
                "CUDA HF GatedDeltaNet-MoE descriptor path cannot source incompatible tensor {}",
                entry.name
            ),
        }),
        _ => unreachable!("static roles handled before layer source lookup"),
    }
}

fn gdn_moe_expert_gate_up_slice<'a>(
    view: PrecisionGatedDeltaNetMoeEncodedView<'a>,
    entry: &HfTensorManifestEntry,
    up: bool,
) -> Result<&'a [u16]> {
    let expert = required_expert_index(entry)? as usize;
    let per_projection = view
        .moe
        .moe_intermediate
        .checked_mul(view.shape.hidden)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: view.moe.moe_intermediate,
            reason: "CUDA HF GDN-MoE expert gate/up source size overflow".to_string(),
        })?;
    let expert_stride =
        per_projection
            .checked_mul(2)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: per_projection,
                reason: "CUDA HF GDN-MoE expert gate/up stride overflow".to_string(),
            })?;
    let start = expert
        .checked_mul(expert_stride)
        .and_then(|base| base.checked_add(if up { per_projection } else { 0 }))
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: expert_stride,
            reason: "CUDA HF GDN-MoE expert gate/up offset overflow".to_string(),
        })?;
    let end = start
        .checked_add(per_projection)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: per_projection,
            reason: "CUDA HF GDN-MoE expert gate/up end overflow".to_string(),
        })?;
    view.expert_gate_up
        .get(start..end)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!(
                "CUDA HF GDN-MoE expert tensor {} is outside gate/up buffer",
                entry.name
            ),
        })
}

fn gdn_moe_expert_down_slice<'a>(
    view: PrecisionGatedDeltaNetMoeEncodedView<'a>,
    entry: &HfTensorManifestEntry,
) -> Result<&'a [u16]> {
    let Some(expert) = entry.expert else {
        return Ok(view.expert_down);
    };
    let expert = expert as usize;
    let per_projection = view
        .shape
        .hidden
        .checked_mul(view.moe.moe_intermediate)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: view.shape.hidden,
            reason: "CUDA HF GDN-MoE expert down source size overflow".to_string(),
        })?;
    let start = expert
        .checked_mul(per_projection)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: per_projection,
            reason: "CUDA HF GDN-MoE expert down offset overflow".to_string(),
        })?;
    let end = start
        .checked_add(per_projection)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: per_projection,
            reason: "CUDA HF GDN-MoE expert down end overflow".to_string(),
        })?;
    view.expert_down
        .get(start..end)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!(
                "CUDA HF GDN-MoE expert tensor {} is outside down buffer",
                entry.name
            ),
        })
}

fn dense_layer_weight_source<'a>(
    view: PrecisionTransformerBlockEncodedView<'a>,
    entry: &HfTensorManifestEntry,
    role: WeightBlockRole,
) -> Result<&'a [u16]> {
    match role {
        WeightBlockRole::AttentionNorm => Ok(view.rms_attn_weight),
        WeightBlockRole::QueryProjection => Ok(view.w_q),
        WeightBlockRole::QueryNorm => required_bias(view.q_norm_weight, &entry.name),
        WeightBlockRole::KeyProjection => Ok(view.w_k),
        WeightBlockRole::KeyNorm => required_bias(view.k_norm_weight, &entry.name),
        WeightBlockRole::ValueProjection => Ok(view.w_v),
        WeightBlockRole::OutputProjection => Ok(view.w_o),
        WeightBlockRole::MlpNorm => Ok(view.rms_mlp_weight),
        WeightBlockRole::GateProjection => Ok(view.w_gate),
        WeightBlockRole::UpProjection => Ok(view.w_up),
        WeightBlockRole::DownProjection => Ok(view.w_down),
        WeightBlockRole::RouterProjection
        | WeightBlockRole::ExpertGateProjection
        | WeightBlockRole::ExpertUpProjection
        | WeightBlockRole::ExpertGateUpProjection
        | WeightBlockRole::ExpertDownProjection
        | WeightBlockRole::SharedExpertGateProjection
        | WeightBlockRole::SharedExpertUpProjection
        | WeightBlockRole::SharedExpertDownProjection
        | WeightBlockRole::SharedExpertRouterProjection => Err(NervaError::InvalidArgument {
            reason: format!(
                "CUDA HF dense descriptor path cannot source MoE tensor {}",
                entry.name
            ),
        }),
        WeightBlockRole::QueryBias => required_bias(view.q_bias, &entry.name),
        WeightBlockRole::KeyBias => required_bias(view.k_bias, &entry.name),
        WeightBlockRole::ValueBias => required_bias(view.v_bias, &entry.name),
        WeightBlockRole::OutputBias => required_bias(view.o_bias, &entry.name),
        _ => unreachable!("static roles handled before layer source lookup"),
    }
}

fn sparse_moe_layer_weight_source<'a>(
    view: PrecisionMoeTransformerBlockEncodedView<'a>,
    entry: &HfTensorManifestEntry,
    role: WeightBlockRole,
) -> Result<&'a [u16]> {
    match role {
        WeightBlockRole::AttentionNorm => Ok(view.rms_attn_weight),
        WeightBlockRole::QueryProjection => Ok(view.w_q),
        WeightBlockRole::QueryNorm => required_bias(view.q_norm_weight, &entry.name),
        WeightBlockRole::KeyProjection => Ok(view.w_k),
        WeightBlockRole::KeyNorm => required_bias(view.k_norm_weight, &entry.name),
        WeightBlockRole::ValueProjection => Ok(view.w_v),
        WeightBlockRole::OutputProjection => Ok(view.w_o),
        WeightBlockRole::MlpNorm => Ok(view.rms_mlp_weight),
        WeightBlockRole::RouterProjection => Ok(view.router),
        WeightBlockRole::ExpertGateProjection => {
            sparse_moe_expert_gate_up_slice(view, entry, false)
        }
        WeightBlockRole::ExpertUpProjection => sparse_moe_expert_gate_up_slice(view, entry, true),
        WeightBlockRole::ExpertGateUpProjection => Ok(view.expert_gate_up),
        WeightBlockRole::ExpertDownProjection => sparse_moe_expert_down_slice(view, entry),
        WeightBlockRole::SharedExpertGateProjection => Ok(view.shared_expert_gate),
        WeightBlockRole::SharedExpertUpProjection => Ok(view.shared_expert_up),
        WeightBlockRole::SharedExpertDownProjection => Ok(view.shared_expert_down),
        WeightBlockRole::SharedExpertRouterProjection => Ok(view.shared_expert_router),
        WeightBlockRole::GateProjection
        | WeightBlockRole::UpProjection
        | WeightBlockRole::DownProjection => Err(NervaError::InvalidArgument {
            reason: format!(
                "CUDA HF sparse MoE descriptor path cannot source dense MLP tensor {}",
                entry.name
            ),
        }),
        WeightBlockRole::QueryBias => required_bias(view.q_bias, &entry.name),
        WeightBlockRole::KeyBias => required_bias(view.k_bias, &entry.name),
        WeightBlockRole::ValueBias => required_bias(view.v_bias, &entry.name),
        WeightBlockRole::OutputBias => required_bias(view.o_bias, &entry.name),
        _ => unreachable!("static roles handled before layer source lookup"),
    }
}

fn sparse_moe_expert_gate_up_slice<'a>(
    view: PrecisionMoeTransformerBlockEncodedView<'a>,
    entry: &HfTensorManifestEntry,
    up: bool,
) -> Result<&'a [u16]> {
    let expert = required_expert_index(entry)? as usize;
    let per_projection = view
        .moe_intermediate
        .checked_mul(view.shape.hidden)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: view.moe_intermediate,
            reason: "CUDA HF MoE expert gate/up source size overflow".to_string(),
        })?;
    let expert_stride =
        per_projection
            .checked_mul(2)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: per_projection,
                reason: "CUDA HF MoE expert gate/up stride overflow".to_string(),
            })?;
    let start = expert
        .checked_mul(expert_stride)
        .and_then(|base| base.checked_add(if up { per_projection } else { 0 }))
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: expert_stride,
            reason: "CUDA HF MoE expert gate/up offset overflow".to_string(),
        })?;
    let end = start
        .checked_add(per_projection)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: per_projection,
            reason: "CUDA HF MoE expert gate/up end overflow".to_string(),
        })?;
    view.expert_gate_up
        .get(start..end)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!(
                "CUDA HF MoE expert tensor {} is outside gate/up buffer",
                entry.name
            ),
        })
}

fn sparse_moe_expert_down_slice<'a>(
    view: PrecisionMoeTransformerBlockEncodedView<'a>,
    entry: &HfTensorManifestEntry,
) -> Result<&'a [u16]> {
    let Some(expert) = entry.expert else {
        return Ok(view.expert_down);
    };
    let expert = expert as usize;
    let per_projection = view
        .shape
        .hidden
        .checked_mul(view.moe_intermediate)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: view.shape.hidden,
            reason: "CUDA HF MoE expert down source size overflow".to_string(),
        })?;
    let start = expert
        .checked_mul(per_projection)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: per_projection,
            reason: "CUDA HF MoE expert down offset overflow".to_string(),
        })?;
    let end = start
        .checked_add(per_projection)
        .ok_or_else(|| NervaError::AllocationFailed {
            bytes: per_projection,
            reason: "CUDA HF MoE expert down end overflow".to_string(),
        })?;
    view.expert_down
        .get(start..end)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!(
                "CUDA HF MoE expert tensor {} is outside down buffer",
                entry.name
            ),
        })
}

fn required_expert_index(entry: &HfTensorManifestEntry) -> Result<u32> {
    entry.expert.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!(
            "CUDA HF MoE expert tensor {} has no expert index",
            entry.name
        ),
    })
}

fn required_bias<'a>(bias: Option<&'a [u16]>, name: &str) -> Result<&'a [u16]> {
    bias.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("CUDA HF descriptor source {name} is missing"),
    })
}

pub(super) fn cuda_weight_strategy(strategy: ResidentWeightExecutionStrategy) -> Result<u32> {
    match strategy {
        ResidentWeightExecutionStrategy::GpuResident => Ok(CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT),
        ResidentWeightExecutionStrategy::GpuStaged => Ok(CUDA_HF_WEIGHT_STRATEGY_GPU_STAGED),
        other => Err(NervaError::InvalidArgument {
            reason: format!(
                "CUDA HF decode cannot consume resident weight strategy {}",
                other.as_str()
            ),
        }),
    }
}
