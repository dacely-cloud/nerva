mod names;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::memory::tier::MemoryTier;

use crate::common::json::format::json_opt_str;
use crate::hf::architecture::HfArchitectureKind;
use crate::weights::hash::hash_tensor_manifest;
use crate::weights::layout::entry::{WeightBlockRole, WeightBlockSpec};
use crate::weights::layout::plan::HfWeightLayoutPlan;
use crate::weights::layout::probe::hf_weight_layout_probe;
use crate::weights::manifest::names::{
    ensure_supported_hf_tensor_names, hf_expert_tensor_name, hf_tensor_name, weight_block_rank,
};

#[derive(Clone, Debug, PartialEq)]
pub struct HfTensorManifestEntry {
    pub name: String,
    pub role: WeightBlockRole,
    pub layer: Option<u32>,
    pub expert: Option<u32>,
    pub rows: usize,
    pub cols: usize,
    pub depth: Option<usize>,
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
            expert: None,
            rows: block.rows,
            cols: block.cols,
            depth: block.depth,
            rank: weight_block_rank(block.role),
            elements: block.elements,
            bytes: block.bytes,
            dtype: block.dtype,
            tier: block.tier,
        }
    }

    fn from_parts(
        name: String,
        role: WeightBlockRole,
        layer: Option<u32>,
        expert: Option<u32>,
        rows: usize,
        cols: usize,
        rank: u8,
        dtype: DType,
        tier: MemoryTier,
    ) -> Result<Self> {
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: 0,
                reason: format!(
                    "HF tensor manifest {} element count overflow",
                    role.as_str()
                ),
            })?;
        let bytes = dtype
            .packed_storage_bytes(elements)
            .map_err(|err| match err {
                NervaError::AllocationFailed { bytes, reason } => NervaError::AllocationFailed {
                    bytes,
                    reason: format!("HF tensor manifest {} {reason}", role.as_str()),
                },
                other => other,
            })?;
        Ok(Self {
            name,
            role,
            layer,
            expert,
            rows,
            cols,
            depth: None,
            rank,
            elements,
            bytes,
            dtype,
            tier,
        })
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
        push_manifest_entries_for_block(&mut entries, plan.metadata.architecture, block)?;
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

fn push_manifest_entries_for_block(
    entries: &mut Vec<HfTensorManifestEntry>,
    architecture: HfArchitectureKind,
    block: WeightBlockSpec,
) -> Result<()> {
    if uses_split_expert_tensors(architecture) {
        match block.role {
            WeightBlockRole::ExpertGateUpProjection => {
                return push_qwen_moe_expert_gate_up_entries(entries, architecture, block);
            }
            WeightBlockRole::ExpertDownProjection => {
                return push_qwen_moe_expert_down_entries(entries, architecture, block);
            }
            _ => {}
        }
    }
    if uses_deepseek_v3_expert_tensors(architecture) {
        match block.role {
            WeightBlockRole::ExpertGateProjection
            | WeightBlockRole::ExpertUpProjection
            | WeightBlockRole::ExpertDownProjection
            | WeightBlockRole::ExpertGateScaleInv
            | WeightBlockRole::ExpertUpScaleInv
            | WeightBlockRole::ExpertDownScaleInv => {
                return push_deepseek_expert_entries(entries, architecture, block);
            }
            _ => {}
        }
    }
    if uses_deepseek_v4_expert_tensors(architecture) {
        match block.role {
            WeightBlockRole::ExpertGateProjection
            | WeightBlockRole::ExpertUpProjection
            | WeightBlockRole::ExpertDownProjection
            | WeightBlockRole::DeepSeekV4ExpertGateScale
            | WeightBlockRole::DeepSeekV4ExpertUpScale
            | WeightBlockRole::DeepSeekV4ExpertDownScale => {
                return push_deepseek_expert_entries(entries, architecture, block);
            }
            _ => {}
        }
    }
    let name = hf_tensor_name(architecture, block.role, block.layer)?;
    entries.push(HfTensorManifestEntry::from_block(block, name));
    Ok(())
}

fn uses_split_expert_tensors(architecture: HfArchitectureKind) -> bool {
    matches!(
        architecture,
        HfArchitectureKind::MixtralMoe
            | HfArchitectureKind::Qwen2Moe
            | HfArchitectureKind::Qwen3Moe
    )
}

fn uses_deepseek_v3_expert_tensors(architecture: HfArchitectureKind) -> bool {
    matches!(
        architecture,
        HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32
    )
}

fn uses_deepseek_v4_expert_tensors(architecture: HfArchitectureKind) -> bool {
    architecture == HfArchitectureKind::DeepSeekV4
}

fn push_deepseek_expert_entries(
    entries: &mut Vec<HfTensorManifestEntry>,
    architecture: HfArchitectureKind,
    block: WeightBlockSpec,
) -> Result<()> {
    let experts = block.depth.ok_or_else(|| NervaError::InvalidArgument {
        reason: "DeepSeek expert block is missing depth".to_string(),
    })?;
    for expert in 0..experts {
        let expert = u32::try_from(expert).map_err(|_| NervaError::InvalidArgument {
            reason: "DeepSeek expert index does not fit u32".to_string(),
        })?;
        let name = hf_expert_tensor_name(architecture, block.role, block.layer, expert)?;
        entries.push(HfTensorManifestEntry::from_parts(
            name,
            block.role,
            block.layer,
            Some(expert),
            block.rows,
            block.cols,
            2,
            block.dtype,
            block.tier,
        )?);
    }
    Ok(())
}

fn push_qwen_moe_expert_gate_up_entries(
    entries: &mut Vec<HfTensorManifestEntry>,
    architecture: HfArchitectureKind,
    block: WeightBlockSpec,
) -> Result<()> {
    let experts = block.depth.ok_or_else(|| NervaError::InvalidArgument {
        reason: "Qwen MoE expert gate/up block is missing depth".to_string(),
    })?;
    if block.rows % 2 != 0 {
        return Err(NervaError::InvalidArgument {
            reason: "Qwen MoE expert gate/up block rows must be even".to_string(),
        });
    }
    let rows = block.rows / 2;
    for expert in 0..experts {
        let expert = u32::try_from(expert).map_err(|_| NervaError::InvalidArgument {
            reason: "Qwen MoE expert index does not fit u32".to_string(),
        })?;
        for role in [
            WeightBlockRole::ExpertGateProjection,
            WeightBlockRole::ExpertUpProjection,
        ] {
            let name = hf_expert_tensor_name(architecture, role, block.layer, expert)?;
            entries.push(HfTensorManifestEntry::from_parts(
                name,
                role,
                block.layer,
                Some(expert),
                rows,
                block.cols,
                2,
                block.dtype,
                block.tier,
            )?);
        }
    }
    Ok(())
}

fn push_qwen_moe_expert_down_entries(
    entries: &mut Vec<HfTensorManifestEntry>,
    architecture: HfArchitectureKind,
    block: WeightBlockSpec,
) -> Result<()> {
    let experts = block.depth.ok_or_else(|| NervaError::InvalidArgument {
        reason: "Qwen MoE expert down block is missing depth".to_string(),
    })?;
    for expert in 0..experts {
        let expert = u32::try_from(expert).map_err(|_| NervaError::InvalidArgument {
            reason: "Qwen MoE expert index does not fit u32".to_string(),
        })?;
        let name = hf_expert_tensor_name(
            architecture,
            WeightBlockRole::ExpertDownProjection,
            block.layer,
            expert,
        )?;
        entries.push(HfTensorManifestEntry::from_parts(
            name,
            WeightBlockRole::ExpertDownProjection,
            block.layer,
            Some(expert),
            block.rows,
            block.cols,
            2,
            block.dtype,
            block.tier,
        )?);
    }
    Ok(())
}

pub fn hf_tensor_manifest_probe() -> Result<HfTensorManifestProbeSummary> {
    let plan = hf_weight_layout_probe()?.plan;
    let manifest = build_hf_tensor_manifest(&plan)?;
    Ok(HfTensorManifestProbeSummary {
        status: HfTensorManifestProbeStatus::Ok,
        manifest,
    })
}
