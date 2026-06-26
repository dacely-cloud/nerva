use nerva_core::types::{DType, MemoryTier, NervaError, Result};

use crate::common::json::json_opt_str;
use crate::hf::architecture::HfArchitectureKind;
use crate::weights::hash::hash_tensor_manifest;
use crate::weights::layout::{
    HfWeightLayoutPlan, WeightBlockRole, WeightBlockSpec, hf_weight_layout_probe,
};

#[derive(Clone, Debug, PartialEq)]
pub struct HfTensorManifestEntry {
    pub name: String,
    pub role: WeightBlockRole,
    pub layer: Option<u32>,
    pub rows: usize,
    pub cols: usize,
    pub rank: u8,
    pub elements: usize,
    pub bytes: usize,
    pub dtype: DType,
    pub tier: MemoryTier,
}

impl HfTensorManifestEntry {
    fn from_block(block: WeightBlockSpec, name: String) -> Self {
        Self {
            name,
            role: block.role,
            layer: block.layer,
            rows: block.rows,
            cols: block.cols,
            rank: weight_block_rank(block.role),
            elements: block.elements,
            bytes: block.bytes,
            dtype: block.dtype,
            tier: block.tier,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfTensorManifest {
    pub architecture: HfArchitectureKind,
    pub entries: Vec<HfTensorManifestEntry>,
    pub total_weight_bytes: usize,
    pub manifest_hash: u64,
}

impl HfTensorManifest {
    pub fn to_json(&self) -> String {
        let first = self.entries.first().map(|entry| entry.name.as_str());
        let last = self.entries.last().map(|entry| entry.name.as_str());
        format!(
            "{{\"architecture\":\"{}\",\"entries\":{},\"total_weight_bytes\":{},\"first_tensor\":{},\"last_tensor\":{},\"manifest_hash\":{}}}",
            self.architecture.as_str(),
            self.entries.len(),
            self.total_weight_bytes,
            json_opt_str(first),
            json_opt_str(last),
            self.manifest_hash,
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HfTensorManifestProbeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HfTensorManifestProbeSummary {
    pub status: HfTensorManifestProbeStatus,
    pub manifest: HfTensorManifest,
}

impl HfTensorManifestProbeSummary {
    pub fn to_json(&self) -> String {
        let status = match self.status {
            HfTensorManifestProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"manifest\":{}}}",
            status,
            self.manifest.to_json(),
        )
    }
}

pub fn build_hf_tensor_manifest(plan: &HfWeightLayoutPlan) -> Result<HfTensorManifest> {
    ensure_supported_hf_tensor_names(plan.metadata.architecture)?;
    let mut entries = Vec::with_capacity(plan.blocks.len());
    for block in plan.blocks.iter().copied() {
        let name = hf_tensor_name(plan.metadata.architecture, block.role, block.layer)?;
        entries.push(HfTensorManifestEntry::from_block(block, name));
    }
    let total_weight_bytes = entries.iter().try_fold(0usize, |acc, entry| {
        acc.checked_add(entry.bytes)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: entry.bytes,
                reason: "HF tensor manifest byte count overflow".to_string(),
            })
    })?;
    if total_weight_bytes != plan.total_weight_bytes {
        return Err(NervaError::InvalidArgument {
            reason: "HF tensor manifest byte count does not match layout plan".to_string(),
        });
    }

    let mut manifest = HfTensorManifest {
        architecture: plan.metadata.architecture,
        entries,
        total_weight_bytes,
        manifest_hash: 0,
    };
    manifest.manifest_hash = hash_tensor_manifest(&manifest);
    Ok(manifest)
}

pub fn hf_tensor_manifest_probe() -> Result<HfTensorManifestProbeSummary> {
    let plan = hf_weight_layout_probe()?.plan;
    let manifest = build_hf_tensor_manifest(&plan)?;
    Ok(HfTensorManifestProbeSummary {
        status: HfTensorManifestProbeStatus::Ok,
        manifest,
    })
}

pub(crate) fn ensure_supported_hf_tensor_names(architecture: HfArchitectureKind) -> Result<()> {
    match architecture {
        HfArchitectureKind::Llama | HfArchitectureKind::Mistral | HfArchitectureKind::Qwen2 => {
            Ok(())
        }
        HfArchitectureKind::Gemma | HfArchitectureKind::Unknown => {
            Err(NervaError::InvalidArgument {
                reason: format!(
                    "HF tensor names for architecture {} are not implemented",
                    architecture.as_str()
                ),
            })
        }
    }
}

pub(crate) fn hf_tensor_name(
    architecture: HfArchitectureKind,
    role: WeightBlockRole,
    layer: Option<u32>,
) -> Result<String> {
    ensure_supported_hf_tensor_names(architecture)?;
    match role {
        WeightBlockRole::TokenEmbedding => {
            require_static_tensor(role, layer).map(|()| "model.embed_tokens.weight".to_string())
        }
        WeightBlockRole::LmHead => {
            require_static_tensor(role, layer).map(|()| "lm_head.weight".to_string())
        }
        WeightBlockRole::AttentionNorm => layer_name(role, layer, "input_layernorm.weight"),
        WeightBlockRole::MlpNorm => layer_name(role, layer, "post_attention_layernorm.weight"),
        WeightBlockRole::QueryProjection => layer_name(role, layer, "self_attn.q_proj.weight"),
        WeightBlockRole::KeyProjection => layer_name(role, layer, "self_attn.k_proj.weight"),
        WeightBlockRole::ValueProjection => layer_name(role, layer, "self_attn.v_proj.weight"),
        WeightBlockRole::OutputProjection => layer_name(role, layer, "self_attn.o_proj.weight"),
        WeightBlockRole::GateProjection => layer_name(role, layer, "mlp.gate_proj.weight"),
        WeightBlockRole::UpProjection => layer_name(role, layer, "mlp.up_proj.weight"),
        WeightBlockRole::DownProjection => layer_name(role, layer, "mlp.down_proj.weight"),
    }
}

pub(crate) fn require_static_tensor(role: WeightBlockRole, layer: Option<u32>) -> Result<()> {
    if layer.is_none() {
        Ok(())
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!("weight block {} must not have a layer", role.as_str()),
        })
    }
}

pub(crate) fn layer_name(
    role: WeightBlockRole,
    layer: Option<u32>,
    suffix: &'static str,
) -> Result<String> {
    let layer = layer.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("weight block {} must have a layer", role.as_str()),
    })?;
    Ok(format!("model.layers.{layer}.{suffix}"))
}

pub(crate) fn weight_block_rank(role: WeightBlockRole) -> u8 {
    match role {
        WeightBlockRole::AttentionNorm | WeightBlockRole::MlpNorm => 1,
        WeightBlockRole::TokenEmbedding
        | WeightBlockRole::QueryProjection
        | WeightBlockRole::KeyProjection
        | WeightBlockRole::ValueProjection
        | WeightBlockRole::OutputProjection
        | WeightBlockRole::GateProjection
        | WeightBlockRole::UpProjection
        | WeightBlockRole::DownProjection
        | WeightBlockRole::LmHead => 2,
    }
}
