use nerva_core::types::error::{NervaError, Result};

use crate::common::shape::TransformerBlockShape;
use crate::precision::bits::f32_to_f16_bits;
use crate::precision::file_smoke::constants::SHARD_NAME;
use crate::reference::block::ReferenceTransformerBlock;
use crate::weights::layout::WeightBlockRole;

pub(crate) fn tiny_file_block_manifest() -> Result<crate::weights::manifest::HfTensorManifest> {
    let metadata = crate::hf::parser::parse_hf_config_metadata(
        r#"{
                "model_type": "llama",
                "hidden_size": 2,
                "intermediate_size": 2,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 4,
                "torch_dtype": "float16"
            }"#,
    )?;
    let layout = crate::weights::layout::plan_hf_weight_layout(&metadata)?;
    crate::weights::manifest::build_hf_tensor_manifest(&layout)
}

pub(crate) fn tensor_payload_for_manifest(
    manifest: &crate::weights::manifest::HfTensorManifest,
) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    for entry in &manifest.entries {
        let values = tensor_values_for_entry(entry)?;
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(payload)
}

fn tensor_values_for_entry(
    entry: &crate::weights::manifest::HfTensorManifestEntry,
) -> Result<Vec<u16>> {
    let elements = entry.bytes / 2;
    let values = match entry.role {
        WeightBlockRole::AttentionNorm | WeightBlockRole::MlpNorm => {
            vec![f32_to_f16_bits(1.0); elements]
        }
        WeightBlockRole::QueryProjection
        | WeightBlockRole::KeyProjection
        | WeightBlockRole::ValueProjection
        | WeightBlockRole::OutputProjection
        | WeightBlockRole::UpProjection
        | WeightBlockRole::DownProjection => encoded_identity(entry.rows, entry.cols, 1.0),
        WeightBlockRole::GateProjection => encoded_identity(entry.rows, entry.cols, 0.5),
        WeightBlockRole::TokenEmbedding | WeightBlockRole::LmHead => vec![0; elements],
    };
    if values.len() == elements {
        Ok(values)
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!("synthetic tensor {} has wrong element count", entry.name),
        })
    }
}

fn encoded_identity(rows: usize, cols: usize, diagonal: f32) -> Vec<u16> {
    let mut values = vec![0u16; rows * cols];
    let encoded = f32_to_f16_bits(diagonal);
    for index in 0..rows.min(cols) {
        values[index * cols + index] = encoded;
    }
    values
}

pub(crate) fn single_shard_index_json(
    manifest: &crate::weights::manifest::HfTensorManifest,
) -> String {
    let mut out = format!(
        "{{\"metadata\":{{\"total_size\":{}}},\"weight_map\":{{",
        manifest.total_weight_bytes
    );
    for (index, entry) in manifest.entries.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&entry.name);
        out.push_str("\":\"");
        out.push_str(SHARD_NAME);
        out.push('"');
    }
    out.push_str("}}");
    out
}

pub(crate) fn reference_block(shape: TransformerBlockShape) -> Result<ReferenceTransformerBlock> {
    ReferenceTransformerBlock::new(
        shape,
        vec![1.0, 1.0],
        vec![1.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![0.5, 0.0, 0.0, 0.5],
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        1e-5,
    )
}
