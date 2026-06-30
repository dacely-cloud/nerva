use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};

use crate::common::shape::TransformerBlockShape;
use crate::precision::bits::f32_to_f16_bits;
use crate::precision::file_smoke::constants::SHARD_NAME;
use crate::reference::block::types::ReferenceTransformerBlock;
use crate::weights::layout::entry::WeightBlockRole;

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
    let layout = crate::weights::layout::plan::plan_hf_weight_layout(&metadata)?;
    crate::weights::manifest::build_hf_tensor_manifest(&layout)
}

pub(crate) fn tensor_payload_for_manifest(
    manifest: &crate::weights::manifest::HfTensorManifest,
) -> Result<Vec<u8>> {
    let mut payload = Vec::new();
    for entry in &manifest.entries {
        payload.extend_from_slice(&tensor_bytes_for_entry(entry)?);
    }
    Ok(payload)
}

fn tensor_bytes_for_entry(
    entry: &crate::weights::manifest::HfTensorManifestEntry,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(entry.bytes);
    match entry.dtype {
        DType::F32 => {
            let value = match entry.role {
                WeightBlockRole::AttentionNorm
                | WeightBlockRole::QueryNorm
                | WeightBlockRole::KeyNorm
                | WeightBlockRole::LinearNorm
                | WeightBlockRole::MlpNorm
                | WeightBlockRole::FinalNorm => 1.0f32,
                _ => 0.0f32,
            };
            for _ in 0..entry.elements {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        DType::F16 | DType::BF16 => {
            for value in tensor_values_for_entry(entry)? {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        _ => {
            return Err(NervaError::InvalidArgument {
                reason: format!("synthetic tensor {} has unsupported dtype", entry.name),
            });
        }
    }
    if bytes.len() == entry.bytes {
        Ok(bytes)
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!("synthetic tensor {} has wrong byte count", entry.name),
        })
    }
}

fn tensor_values_for_entry(
    entry: &crate::weights::manifest::HfTensorManifestEntry,
) -> Result<Vec<u16>> {
    let elements = entry.bytes / 2;
    let values = match entry.role {
        WeightBlockRole::AttentionNorm
        | WeightBlockRole::QueryNorm
        | WeightBlockRole::KeyNorm
        | WeightBlockRole::DeepSeekQALoraNorm
        | WeightBlockRole::DeepSeekKvANorm
        | WeightBlockRole::DeepSeekIndexerKeyNorm
        | WeightBlockRole::DeepSeekV4QNorm
        | WeightBlockRole::DeepSeekV4KvNorm
        | WeightBlockRole::DeepSeekV4CompressorNorm
        | WeightBlockRole::DeepSeekV4IndexerCompressorNorm
        | WeightBlockRole::LinearNorm
        | WeightBlockRole::MlpNorm
        | WeightBlockRole::FinalNorm => vec![f32_to_f16_bits(1.0); elements],
        WeightBlockRole::QueryProjection
        | WeightBlockRole::KeyProjection
        | WeightBlockRole::ValueProjection
        | WeightBlockRole::OutputProjection
        | WeightBlockRole::DeepSeekQALoraProjection
        | WeightBlockRole::DeepSeekQBProjection
        | WeightBlockRole::DeepSeekKvAProjection
        | WeightBlockRole::DeepSeekKvBProjection
        | WeightBlockRole::DeepSeekIndexerQueryProjection
        | WeightBlockRole::DeepSeekIndexerKeyProjection
        | WeightBlockRole::DeepSeekIndexerWeightsProjection
        | WeightBlockRole::DeepSeekV4WqAProjection
        | WeightBlockRole::DeepSeekV4WqBProjection
        | WeightBlockRole::DeepSeekV4WkvProjection
        | WeightBlockRole::DeepSeekV4WoAProjection
        | WeightBlockRole::DeepSeekV4WoBProjection
        | WeightBlockRole::DeepSeekV4CompressorWkvProjection
        | WeightBlockRole::DeepSeekV4CompressorWgateProjection
        | WeightBlockRole::DeepSeekV4IndexerWqBProjection
        | WeightBlockRole::DeepSeekV4IndexerCompressorWkvProjection
        | WeightBlockRole::DeepSeekV4IndexerCompressorWgateProjection
        | WeightBlockRole::DeepSeekV4IndexerWeightsProjection
        | WeightBlockRole::LinearQkvProjection
        | WeightBlockRole::LinearZProjection
        | WeightBlockRole::LinearOutputProjection
        | WeightBlockRole::UpProjection
        | WeightBlockRole::DownProjection => encoded_identity(entry.rows, entry.cols, 1.0),
        WeightBlockRole::GateProjection => encoded_identity(entry.rows, entry.cols, 0.5),
        WeightBlockRole::QueryBias
        | WeightBlockRole::KeyBias
        | WeightBlockRole::ValueBias
        | WeightBlockRole::OutputBias
        | WeightBlockRole::DeepSeekQALoraScaleInv
        | WeightBlockRole::DeepSeekQBScaleInv
        | WeightBlockRole::DeepSeekKvAScaleInv
        | WeightBlockRole::DeepSeekKvBScaleInv
        | WeightBlockRole::DeepSeekOutputScaleInv
        | WeightBlockRole::DeepSeekIndexerQueryScaleInv
        | WeightBlockRole::DeepSeekIndexerKeyScaleInv
        | WeightBlockRole::DeepSeekIndexerKeyNormBias
        | WeightBlockRole::DeepSeekV4HcHeadBase
        | WeightBlockRole::DeepSeekV4HcHeadFn
        | WeightBlockRole::DeepSeekV4HcHeadScale
        | WeightBlockRole::DeepSeekV4HcAttnBase
        | WeightBlockRole::DeepSeekV4HcAttnFn
        | WeightBlockRole::DeepSeekV4HcAttnScale
        | WeightBlockRole::DeepSeekV4HcFfnBase
        | WeightBlockRole::DeepSeekV4HcFfnFn
        | WeightBlockRole::DeepSeekV4HcFfnScale
        | WeightBlockRole::DeepSeekV4AttentionSink
        | WeightBlockRole::DeepSeekV4WqAScale
        | WeightBlockRole::DeepSeekV4WqBScale
        | WeightBlockRole::DeepSeekV4WkvScale
        | WeightBlockRole::DeepSeekV4WoAScale
        | WeightBlockRole::DeepSeekV4WoBScale
        | WeightBlockRole::DeepSeekV4CompressorApe
        | WeightBlockRole::DeepSeekV4CompressorWkvScale
        | WeightBlockRole::DeepSeekV4CompressorWgateScale
        | WeightBlockRole::DeepSeekV4IndexerWqBScale
        | WeightBlockRole::DeepSeekV4IndexerCompressorApe
        | WeightBlockRole::DeepSeekV4IndexerCompressorWkvScale
        | WeightBlockRole::DeepSeekV4IndexerCompressorWgateScale
        | WeightBlockRole::DeepSeekV4HashRouteTable
        | WeightBlockRole::DeepSeekV4IndexerWeightsScale
        | WeightBlockRole::LinearConvProjection
        | WeightBlockRole::LinearBProjection
        | WeightBlockRole::LinearAProjection
        | WeightBlockRole::LinearDtBias
        | WeightBlockRole::LinearALog
        | WeightBlockRole::TokenEmbedding
        | WeightBlockRole::LmHead
        | WeightBlockRole::GateScaleInv
        | WeightBlockRole::UpScaleInv
        | WeightBlockRole::DownScaleInv
        | WeightBlockRole::RouterProjection
        | WeightBlockRole::RouterCorrectionBias
        | WeightBlockRole::ExpertGateProjection
        | WeightBlockRole::ExpertUpProjection
        | WeightBlockRole::ExpertGateUpProjection
        | WeightBlockRole::ExpertDownProjection
        | WeightBlockRole::ExpertGateScaleInv
        | WeightBlockRole::ExpertUpScaleInv
        | WeightBlockRole::ExpertDownScaleInv
        | WeightBlockRole::SharedExpertGateProjection
        | WeightBlockRole::SharedExpertUpProjection
        | WeightBlockRole::SharedExpertDownProjection
        | WeightBlockRole::SharedExpertGateScaleInv
        | WeightBlockRole::SharedExpertUpScaleInv
        | WeightBlockRole::SharedExpertDownScaleInv
        | WeightBlockRole::DeepSeekV4ExpertGateScale
        | WeightBlockRole::DeepSeekV4ExpertUpScale
        | WeightBlockRole::DeepSeekV4ExpertDownScale
        | WeightBlockRole::DeepSeekV4SharedExpertGateScale
        | WeightBlockRole::DeepSeekV4SharedExpertUpScale
        | WeightBlockRole::DeepSeekV4SharedExpertDownScale
        | WeightBlockRole::SharedExpertRouterProjection => vec![0; elements],
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
