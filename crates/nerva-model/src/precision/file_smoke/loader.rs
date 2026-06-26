use std::path::Path;

use nerva_core::types::error::{NervaError, Result};

use crate::common::hash::hash_bytes;
use crate::weights::layout::WeightBlockRole;
use crate::weights::safetensors::{SafetensorsShardPlan, SafetensorsShardPlanEntry};
use crate::weights::tensor::read_safetensors_tensor_u16;

pub(crate) fn load_role(
    plan: &SafetensorsShardPlan,
    shard_path: &Path,
    role: WeightBlockRole,
) -> Result<LoadedTensor> {
    let entry = plan
        .entries
        .iter()
        .find(|entry| entry.role == role && entry.layer == Some(0))
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!("safetensors plan missing role {:?}", role),
        })?;
    load_entry(shard_path, entry)
}

fn load_entry(shard_path: &Path, entry: &SafetensorsShardPlanEntry) -> Result<LoadedTensor> {
    let tensor = read_safetensors_tensor_u16(shard_path, entry)?;
    Ok(LoadedTensor {
        values: tensor.values,
        bytes_read: tensor.bytes_read,
        data_hash: tensor.data_hash,
    })
}

#[derive(Clone, Debug)]
pub(crate) struct LoadedTensor {
    pub(crate) values: Vec<u16>,
    bytes_read: usize,
    data_hash: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct LoadedBlockWeights {
    pub(crate) rms_attn_weight: LoadedTensor,
    pub(crate) rms_mlp_weight: LoadedTensor,
    pub(crate) w_q: LoadedTensor,
    pub(crate) w_k: LoadedTensor,
    pub(crate) w_v: LoadedTensor,
    pub(crate) w_o: LoadedTensor,
    pub(crate) w_gate: LoadedTensor,
    pub(crate) w_up: LoadedTensor,
    pub(crate) w_down: LoadedTensor,
}

impl LoadedBlockWeights {
    pub(crate) fn bytes_loaded(&self) -> usize {
        self.rms_attn_weight.bytes_read
            + self.rms_mlp_weight.bytes_read
            + self.w_q.bytes_read
            + self.w_k.bytes_read
            + self.w_v.bytes_read
            + self.w_o.bytes_read
            + self.w_gate.bytes_read
            + self.w_up.bytes_read
            + self.w_down.bytes_read
    }

    pub(crate) fn data_hash(&self) -> u64 {
        let mut bytes = Vec::new();
        for hash in [
            self.rms_attn_weight.data_hash,
            self.rms_mlp_weight.data_hash,
            self.w_q.data_hash,
            self.w_k.data_hash,
            self.w_v.data_hash,
            self.w_o.data_hash,
            self.w_gate.data_hash,
            self.w_up.data_hash,
            self.w_down.data_hash,
        ] {
            bytes.extend_from_slice(&hash.to_le_bytes());
        }
        hash_bytes(&bytes)
    }
}
