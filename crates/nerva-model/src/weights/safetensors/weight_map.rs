use std::collections::BTreeMap;

use nerva_core::types::error::{NervaError, Result};

use crate::common::json::fields::optional_usize;
use crate::common::json::parse::find_top_level_json_value;
use crate::weights::safetensors::parse::parse_json_string_map_value;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct SafetensorsWeightMap {
    pub(super) tensor_to_shard: BTreeMap<String, String>,
    pub(super) total_size: Option<usize>,
}

pub(super) fn parse_safetensors_weight_map(index_json: &str) -> Result<SafetensorsWeightMap> {
    let weight_map_json =
        find_top_level_json_value(index_json, "weight_map")?.ok_or_else(|| {
            NervaError::InvalidArgument {
                reason: "safetensors index is missing weight_map".to_string(),
            }
        })?;
    let mut tensor_to_shard = BTreeMap::new();
    for (tensor_name, shard_file) in parse_json_string_map_value(weight_map_json, "weight_map")? {
        if tensor_name.is_empty() || shard_file.is_empty() {
            return Err(NervaError::InvalidArgument {
                reason: "safetensors weight_map entries cannot be empty".to_string(),
            });
        }
        if tensor_to_shard
            .insert(tensor_name.clone(), shard_file)
            .is_some()
        {
            return Err(NervaError::InvalidArgument {
                reason: format!("duplicate safetensors weight_map entry for {tensor_name}"),
            });
        }
    }
    let total_size = match find_top_level_json_value(index_json, "metadata")? {
        Some(metadata_json) => optional_usize(metadata_json, "total_size")?,
        None => None,
    };
    Ok(SafetensorsWeightMap {
        tensor_to_shard,
        total_size,
    })
}
