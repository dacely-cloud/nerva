use std::path::Path;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_ledger::types::token::TokenLedger;

use crate::common::shape::TransformerBlockShape;
use crate::precision::bits::{encode_f32_for_dtype, f32_to_f16_bits, hash_u16s};
use crate::precision::block::PrecisionTransformerBlock;
use crate::precision::file_smoke::fixtures::{
    reference_block, single_shard_index_json, tensor_payload_for_manifest, tiny_file_block_manifest,
};
use crate::precision::file_smoke::loader::{LoadedBlockWeights, load_role};
use crate::precision::file_smoke::{
    PrecisionSafetensorsBlockSmokeStatus, PrecisionSafetensorsBlockSmokeSummary, SHARD_NAME,
};
use crate::precision::scratch::PrecisionTransformerBlockScratch;
use crate::reference::scratch::TransformerBlockScratch;
use crate::weights::layout::WeightBlockRole;
use crate::weights::safetensors::planner::plan_safetensors_shards_for_manifest;
use crate::weights::safetensors::{SafetensorsShardHeader, SafetensorsShardPlan};

pub fn precision_block_from_safetensors_smoke() -> Result<PrecisionSafetensorsBlockSmokeSummary> {
    let shape = TransformerBlockShape::new(2, 1, 2);
    let manifest = tiny_file_block_manifest()?;
    let header = crate::weights::safetensors::synthetic_safetensors_header_for_manifest(&manifest)?;
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
