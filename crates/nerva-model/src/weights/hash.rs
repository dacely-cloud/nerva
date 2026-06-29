use crate::common::dtype::dtype_to_str;
use crate::hf::hash::hash_metadata;
use crate::weights::layout::plan::HfWeightLayoutPlan;
use crate::weights::manifest::HfTensorManifest;
use crate::weights::safetensors::shard::SafetensorsShardPlan;

pub(crate) fn hash_weight_layout(plan: &HfWeightLayoutPlan) -> u64 {
    let mut hash = hash_metadata(&plan.metadata);
    for block in &plan.blocks {
        for byte in block.role.as_str().as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for value in [
            block.layer.map(u64::from).unwrap_or(u64::MAX),
            block.rows as u64,
            block.cols as u64,
            block.depth.unwrap_or(0) as u64,
            block.elements as u64,
            block.bytes as u64,
        ] {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    for value in [
        plan.total_weight_bytes as u64,
        plan.per_layer_weight_bytes as u64,
        plan.static_weight_bytes as u64,
    ] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}
pub(crate) fn hash_tensor_manifest(manifest: &HfTensorManifest) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for entry in &manifest.entries {
        for byte in entry.name.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for byte in entry.role.as_str().as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for value in [
            entry.layer.map(u64::from).unwrap_or(u64::MAX),
            entry.expert.map(u64::from).unwrap_or(u64::MAX),
            entry.rows as u64,
            entry.cols as u64,
            entry.depth.unwrap_or(0) as u64,
            u64::from(entry.rank),
            entry.elements as u64,
            entry.bytes as u64,
        ] {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    for value in [
        manifest.entries.len() as u64,
        manifest.total_weight_bytes as u64,
    ] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}
pub(crate) fn hash_safetensors_shard_plan(plan: &SafetensorsShardPlan) -> u64 {
    let mut hash = plan.manifest_hash ^ plan.index_hash;
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    for shard in &plan.shards {
        for byte in shard.file_name.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for value in [
            shard.tensor_count as u64,
            shard.payload_bytes as u64,
            shard.header_bytes as u64,
        ] {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    for entry in &plan.entries {
        for byte in entry.tensor_name.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for byte in entry.shard_file.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for value in [
            entry.layer.map(u64::from).unwrap_or(u64::MAX),
            entry.expert.map(u64::from).unwrap_or(u64::MAX),
            entry.tier as u64,
            entry.bytes as u64,
            entry.data_offset_begin as u64,
            entry.data_offset_end as u64,
            entry.file_offset_begin as u64,
            entry.file_offset_end as u64,
        ] {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        for byte in entry.role.as_str().as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        for byte in dtype_to_str(entry.dtype).as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for value in [
        plan.entries.len() as u64,
        plan.shards.len() as u64,
        plan.total_weight_bytes as u64,
        plan.index_total_size.unwrap_or_default() as u64,
    ] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}
