use std::path::Path;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::token::TokenLedger;

use crate::common::hash::hash_bytes;
use crate::common::shape::TransformerBlockShape;
use crate::precision::bits::{encode_f32_for_dtype, f32_to_f16_bits, hash_u16s};
use crate::precision::block::PrecisionTransformerBlock;
use crate::precision::scratch::PrecisionTransformerBlockScratch;
use crate::reference::block::ReferenceTransformerBlock;
use crate::reference::scratch::TransformerBlockScratch;
use crate::weights::layout::WeightBlockRole;
use crate::weights::safetensors::{
    SafetensorsShardHeader, SafetensorsShardPlan, SafetensorsShardPlanEntry,
    plan_safetensors_shards_for_manifest, synthetic_safetensors_header_for_manifest,
};
use crate::weights::tensor::read_safetensors_tensor_u16;

const SHARD_NAME: &str = "model.safetensors";

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PrecisionSafetensorsBlockSmokeStatus {
    Ok,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PrecisionSafetensorsBlockSmokeSummary {
    pub status: PrecisionSafetensorsBlockSmokeStatus,
    pub dtype: DType,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub tensors_loaded: usize,
    pub bytes_loaded: usize,
    pub data_hash: u64,
    pub output_hash: u64,
    pub expected_hash: u64,
    pub bit_parity: bool,
    pub hot_path_allocations: u64,
}

impl PrecisionSafetensorsBlockSmokeSummary {
    pub fn passed(&self) -> bool {
        self.bit_parity && self.hot_path_allocations == 0 && self.tensors_loaded == 9
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            PrecisionSafetensorsBlockSmokeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"dtype\":\"float16\",\"hidden\":{},\"heads\":{},\"intermediate\":{},\"tensors_loaded\":{},\"bytes_loaded\":{},\"data_hash\":{},\"output_hash\":{},\"expected_hash\":{},\"bit_parity\":{},\"hot_path_allocations\":{}}}",
            status,
            self.hidden,
            self.heads,
            self.intermediate,
            self.tensors_loaded,
            self.bytes_loaded,
            self.data_hash,
            self.output_hash,
            self.expected_hash,
            self.bit_parity,
            self.hot_path_allocations,
        )
    }
}

pub fn precision_block_from_safetensors_smoke() -> Result<PrecisionSafetensorsBlockSmokeSummary> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let manifest = tiny_file_block_manifest()?;
    let header = synthetic_safetensors_header_for_manifest(&manifest)?;
    let index = single_shard_index_json(&manifest);
    let plan = plan_safetensors_shards_for_manifest(
        &index,
        &[SafetensorsShardHeader::new(SHARD_NAME, &header)],
        &manifest,
    )?;
    let payload = tensor_payload_for_manifest(&manifest)?;
    let dir =
        std::env::temp_dir().join(format!("nerva-precision-file-block-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|err| NervaError::InvalidArgument {
        reason: format!("failed to create {}: {err}", dir.display()),
    })?;
    let shard_path = dir.join(SHARD_NAME);
    let mut bytes = Vec::with_capacity(8 + header.len() + payload.len());
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&payload);
    std::fs::write(&shard_path, bytes).map_err(|err| NervaError::InvalidArgument {
        reason: format!("failed to write {}: {err}", shard_path.display()),
    })?;

    let summary = run_loaded_block_smoke(shape, &plan, &shard_path);

    let _ = std::fs::remove_file(&shard_path);
    let _ = std::fs::remove_dir(&dir);

    let summary = summary?;
    if summary.passed() {
        Ok(summary)
    } else {
        Err(NervaError::InvalidArgument {
            reason: "file-loaded FP16 precision block parity failed".to_string(),
        })
    }
}

fn run_loaded_block_smoke(
    shape: TransformerBlockShape,
    plan: &SafetensorsShardPlan,
    shard_path: &Path,
) -> Result<PrecisionSafetensorsBlockSmokeSummary> {
    let loaded = LoadedBlockWeights {
        rms_attn_weight: load_role(plan, shard_path, WeightBlockRole::AttentionNorm)?,
        rms_mlp_weight: load_role(plan, shard_path, WeightBlockRole::MlpNorm)?,
        w_q: load_role(plan, shard_path, WeightBlockRole::QueryProjection)?,
        w_k: load_role(plan, shard_path, WeightBlockRole::KeyProjection)?,
        w_v: load_role(plan, shard_path, WeightBlockRole::ValueProjection)?,
        w_o: load_role(plan, shard_path, WeightBlockRole::OutputProjection)?,
        w_gate: load_role(plan, shard_path, WeightBlockRole::GateProjection)?,
        w_up: load_role(plan, shard_path, WeightBlockRole::UpProjection)?,
        w_down: load_role(plan, shard_path, WeightBlockRole::DownProjection)?,
    };
    let bytes_loaded = loaded.bytes_loaded();
    let data_hash = loaded.data_hash();
    let block = PrecisionTransformerBlock::new_from_encoded(
        DType::F16,
        shape,
        loaded.rms_attn_weight.values,
        loaded.rms_mlp_weight.values,
        loaded.w_q.values,
        loaded.w_k.values,
        loaded.w_v.values,
        loaded.w_o.values,
        loaded.w_gate.values,
        loaded.w_up.values,
        loaded.w_down.values,
        1e-5,
    )?;
    let input_f32 = [1.0, 2.0];
    let input = [f32_to_f16_bits(input_f32[0]), f32_to_f16_bits(input_f32[1])];
    let mut scratch = PrecisionTransformerBlockScratch::new(shape)?;
    let mut output = [0u16; 2];
    let mut ledger = TokenLedger::new(0);
    block.forward_into(&input, &mut scratch, &mut output, &mut ledger)?;
    ledger.require_zero_hot_path_allocations()?;

    let reference = reference_block(shape)?;
    let mut reference_scratch = TransformerBlockScratch::new(shape)?;
    let mut reference_output = [0.0f32; 2];
    let mut reference_ledger = TokenLedger::new(0);
    reference.forward_into(
        &input_f32,
        &mut reference_scratch,
        &mut reference_output,
        &mut reference_ledger,
    )?;
    let expected = [
        encode_f32_for_dtype(reference_output[0], DType::F16)?,
        encode_f32_for_dtype(reference_output[1], DType::F16)?,
    ];

    Ok(PrecisionSafetensorsBlockSmokeSummary {
        status: PrecisionSafetensorsBlockSmokeStatus::Ok,
        dtype: DType::F16,
        hidden: shape.hidden,
        heads: shape.heads,
        intermediate: shape.intermediate,
        tensors_loaded: 9,
        bytes_loaded,
        data_hash,
        output_hash: hash_u16s(&output),
        expected_hash: hash_u16s(&expected),
        bit_parity: output == expected,
        hot_path_allocations: ledger.hot_path_allocations,
    })
}

fn tiny_file_block_manifest() -> Result<crate::weights::manifest::HfTensorManifest> {
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

fn tensor_payload_for_manifest(
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

fn load_role(
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

fn single_shard_index_json(manifest: &crate::weights::manifest::HfTensorManifest) -> String {
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

fn reference_block(shape: TransformerBlockShape) -> Result<ReferenceTransformerBlock> {
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

#[derive(Clone, Debug)]
struct LoadedTensor {
    values: Vec<u16>,
    bytes_read: usize,
    data_hash: u64,
}

#[derive(Clone, Debug)]
struct LoadedBlockWeights {
    rms_attn_weight: LoadedTensor,
    rms_mlp_weight: LoadedTensor,
    w_q: LoadedTensor,
    w_k: LoadedTensor,
    w_v: LoadedTensor,
    w_o: LoadedTensor,
    w_gate: LoadedTensor,
    w_up: LoadedTensor,
    w_down: LoadedTensor,
}

impl LoadedBlockWeights {
    fn bytes_loaded(&self) -> usize {
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

    fn data_hash(&self) -> u64 {
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
