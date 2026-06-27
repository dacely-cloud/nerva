use nerva_core::types::error::{NervaError, Result};
use nerva_cuda::decode::hf_sequence::weight_plan::{
    CUDA_HF_WEIGHT_STRATEGY_GPU_RESIDENT, CUDA_HF_WEIGHT_STRATEGY_GPU_STAGED,
    CudaHfDecodeSequenceWeightBlock,
};
use nerva_model::causal_lm::types::HfCausalLmLoaded;
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
    let layer = loaded
        .model
        .layer(layer_index)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("CUDA HF descriptor layer {layer_index} is unavailable"),
        })?;
    let view = layer.encoded_view();
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
        WeightBlockRole::QueryBias => required_bias(view.q_bias, &entry.name),
        WeightBlockRole::KeyBias => required_bias(view.k_bias, &entry.name),
        WeightBlockRole::ValueBias => required_bias(view.v_bias, &entry.name),
        WeightBlockRole::OutputBias => required_bias(view.o_bias, &entry.name),
        _ => unreachable!("static roles handled before layer source lookup"),
    }
}

fn required_bias<'a>(bias: Option<&'a [u16]>, name: &str) -> Result<&'a [u16]> {
    bias.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("CUDA HF descriptor source {name} is missing"),
    })
}

fn cuda_weight_strategy(strategy: ResidentWeightExecutionStrategy) -> Result<u32> {
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
