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
    ensure_supported_hf_tensor_names, hf_tensor_name, weight_block_rank,
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
