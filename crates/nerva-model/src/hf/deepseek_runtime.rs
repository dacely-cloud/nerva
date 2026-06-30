use nerva_core::types::error::{NervaError, Result};

use crate::hf::architecture::HfArchitectureKind;
use crate::hf::deepseek::plan_deepseek_vllm_kv_cache;
use crate::hf::metadata::{HfMlpLayerKind, HfModelMetadata};
use crate::weights::layout::entry::WeightBlockRole;
use crate::weights::layout::plan::plan_hf_weight_layout;

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekExecutionUnitCoverage {
    pub unit: String,
    pub status: &'static str,
    pub validated_primitives: Vec<String>,
    pub remaining_gaps: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekLayerReport {
    pub moe_layers: usize,
    pub dense_mlp_layers: usize,
    pub v4_swa_layers: usize,
    pub v4_c4_layers: usize,
    pub v4_c128_layers: usize,
    pub v4_indexer_layers: usize,
    pub v4_hash_router_layers: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DeepSeekAttentionExecutionKind {
    V3Mla,
    V32MlaWithIndexer,
    V4SlidingWindowMla,
    V4CompressedMla,
    V4CompressedMlaWithSparseIndexer,
}

impl DeepSeekAttentionExecutionKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::V3Mla => "deepseek_v3_mla",
            Self::V32MlaWithIndexer => "deepseek_v3_2_mla_with_indexer",
            Self::V4SlidingWindowMla => "deepseek_v4_sliding_window_mla",
            Self::V4CompressedMla => "deepseek_v4_compressed_mla",
            Self::V4CompressedMlaWithSparseIndexer => {
                "deepseek_v4_compressed_mla_with_sparse_indexer"
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekLayerExecution {
    pub layer: usize,
    pub attention_kind: DeepSeekAttentionExecutionKind,
    pub compress_ratio: usize,
    pub index_topk: usize,
    pub primary_kv_cache_group: String,
    pub indexer_kv_cache_group: Option<String>,
    pub uses_sliding_window_cache: bool,
    pub uses_sparse_indexer: bool,
    pub uses_compressed_indexer_cache: bool,
    pub uses_compressor: bool,
    pub uses_hash_router: bool,
    pub uses_moe: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekLayerExecutionPlan {
    pub architecture: HfArchitectureKind,
    pub cache_dtype_str: String,
    pub default_block_size: usize,
    pub layers: Vec<DeepSeekLayerExecution>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekLayerWeightContract {
    pub execution: DeepSeekLayerExecution,
    pub roles: Vec<WeightBlockRole>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekRuntimeWeightContract {
    pub architecture: HfArchitectureKind,
    pub layers: Vec<DeepSeekLayerWeightContract>,
}

pub fn deepseek_layer_execution_plan(
    metadata: &HfModelMetadata,
) -> Result<DeepSeekLayerExecutionPlan> {
    if !metadata.architecture.is_deepseek() {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "DeepSeek layer execution planning requires a DeepSeek architecture, got {}",
                metadata.architecture.as_str()
            ),
        });
    }

    let cache_dtype = deepseek_runtime_cache_dtype(metadata.architecture);
    let kv_plan = plan_deepseek_vllm_kv_cache(metadata, cache_dtype)?;
    let layers = match metadata.architecture {
        HfArchitectureKind::DeepSeekV3 => (0..metadata.num_hidden_layers)
            .map(|layer| {
                require_kv_group(&kv_plan, "v3_main_mla")?;
                Ok(layer_execution(
                    metadata,
                    layer,
                    DeepSeekAttentionExecutionKind::V3Mla,
                    1,
                    "v3_main_mla",
                    None,
                    false,
                    false,
                    false,
                    false,
                ))
            })
            .collect::<Result<Vec<_>>>()?,
        HfArchitectureKind::DeepSeekV32 => (0..metadata.num_hidden_layers)
            .map(|layer| {
                require_kv_group(&kv_plan, "v3_2_main_mla")?;
                require_kv_group(&kv_plan, "v3_2_sparse_indexer")?;
                Ok(layer_execution(
                    metadata,
                    layer,
                    DeepSeekAttentionExecutionKind::V32MlaWithIndexer,
                    1,
                    "v3_2_main_mla",
                    Some("v3_2_sparse_indexer"),
                    false,
                    true,
                    true,
                    false,
                ))
            })
            .collect::<Result<Vec<_>>>()?,
        HfArchitectureKind::DeepSeekV4 => (0..metadata.num_hidden_layers)
            .map(|layer| {
                let compress_ratio = metadata.compress_ratios.get(layer).copied().unwrap_or(0);
                let compress_ratio = compress_ratio.max(1);
                match compress_ratio {
                    1 => {
                        require_kv_group(&kv_plan, "v4_swa")?;
                        Ok(layer_execution(
                            metadata,
                            layer,
                            DeepSeekAttentionExecutionKind::V4SlidingWindowMla,
                            compress_ratio,
                            "v4_swa",
                            None,
                            true,
                            false,
                            false,
                            false,
                        ))
                    }
                    4 => {
                        require_kv_group(&kv_plan, "v4_c4_mla")?;
                        require_kv_group(&kv_plan, "v4_c4_mla_indexer")?;
                        Ok(layer_execution(
                            metadata,
                            layer,
                            DeepSeekAttentionExecutionKind::V4CompressedMlaWithSparseIndexer,
                            compress_ratio,
                            "v4_c4_mla",
                            Some("v4_c4_mla_indexer"),
                            false,
                            true,
                            true,
                            true,
                        ))
                    }
                    128 => {
                        require_kv_group(&kv_plan, "v4_c128_mla")?;
                        require_kv_group(&kv_plan, "v4_c128_mla_indexer")?;
                        Ok(layer_execution(
                            metadata,
                            layer,
                            DeepSeekAttentionExecutionKind::V4CompressedMla,
                            compress_ratio,
                            "v4_c128_mla",
                            Some("v4_c128_mla_indexer"),
                            false,
                            false,
                            true,
                            true,
                        ))
                    }
                    other => Err(NervaError::InvalidArgument {
                        reason: format!(
                            "DeepSeek V4 layer {layer} has unsupported compress_ratio {other}; expected 0/1, 4, or 128"
                        ),
                    }),
                }
            })
            .collect::<Result<Vec<_>>>()?,
        _ => unreachable!("DeepSeek architecture checked above"),
    };

    Ok(DeepSeekLayerExecutionPlan {
        architecture: metadata.architecture,
        cache_dtype_str: cache_dtype.to_string(),
        default_block_size: kv_plan.default_block_size,
        layers,
    })
}

pub fn deepseek_runtime_weight_contract(
    metadata: &HfModelMetadata,
) -> Result<DeepSeekRuntimeWeightContract> {
    let execution_plan = deepseek_layer_execution_plan(metadata)?;
    let weight_plan = plan_hf_weight_layout(metadata)?;
    let mut layers = Vec::with_capacity(execution_plan.layers.len());

    for execution in execution_plan.layers {
        let roles = weight_plan
            .blocks
            .iter()
            .filter(|block| block.layer == Some(execution.layer as u32))
            .map(|block| block.role)
            .collect::<Vec<_>>();
        let required_roles = required_roles_for_execution(metadata, &execution);
        for required in required_roles {
            if !roles.contains(&required) {
                return Err(NervaError::InvalidArgument {
                    reason: format!(
                        "DeepSeek layer {} {:?} is missing runtime weight role {}",
                        execution.layer,
                        execution.attention_kind,
                        required.as_str(),
                    ),
                });
            }
        }
        layers.push(DeepSeekLayerWeightContract { execution, roles });
    }

    Ok(DeepSeekRuntimeWeightContract {
        architecture: metadata.architecture,
        layers,
    })
}

fn layer_execution(
    metadata: &HfModelMetadata,
    layer: usize,
    attention_kind: DeepSeekAttentionExecutionKind,
    compress_ratio: usize,
    primary_kv_cache_group: &str,
    indexer_kv_cache_group: Option<&str>,
    uses_sliding_window_cache: bool,
    uses_sparse_indexer: bool,
    uses_compressed_indexer_cache: bool,
    uses_compressor: bool,
) -> DeepSeekLayerExecution {
    DeepSeekLayerExecution {
        layer,
        attention_kind,
        compress_ratio,
        index_topk: metadata.index_topk.unwrap_or(0),
        primary_kv_cache_group: primary_kv_cache_group.to_string(),
        indexer_kv_cache_group: indexer_kv_cache_group.map(str::to_string),
        uses_sliding_window_cache,
        uses_sparse_indexer,
        uses_compressed_indexer_cache,
        uses_compressor,
        uses_hash_router: layer < metadata.num_hash_layers.unwrap_or(0),
        uses_moe: metadata
            .mlp_layer_types
            .get(layer)
            .is_some_and(|kind| *kind == HfMlpLayerKind::SparseMoe),
    }
}

fn required_roles_for_execution(
    metadata: &HfModelMetadata,
    execution: &DeepSeekLayerExecution,
) -> Vec<WeightBlockRole> {
    let mut roles = vec![WeightBlockRole::AttentionNorm];
    match execution.attention_kind {
        DeepSeekAttentionExecutionKind::V3Mla
        | DeepSeekAttentionExecutionKind::V32MlaWithIndexer => {
            push_deepseek_v3_attention_roles(&mut roles);
            if execution.uses_sparse_indexer {
                push_deepseek_v32_indexer_roles(&mut roles);
            }
        }
        DeepSeekAttentionExecutionKind::V4SlidingWindowMla
        | DeepSeekAttentionExecutionKind::V4CompressedMla
        | DeepSeekAttentionExecutionKind::V4CompressedMlaWithSparseIndexer => {
            push_deepseek_v4_attention_roles(&mut roles);
            if execution.uses_compressor {
                push_deepseek_v4_compressor_roles(&mut roles, false);
            }
            if execution.uses_sparse_indexer {
                push_deepseek_v4_sparse_indexer_roles(&mut roles);
                push_deepseek_v4_compressor_roles(&mut roles, true);
            }
        }
    }

    roles.push(WeightBlockRole::MlpNorm);
    if execution.uses_moe {
        push_deepseek_moe_roles(metadata, execution.layer, &mut roles);
    } else {
        push_deepseek_dense_mlp_roles(&mut roles);
    }
    roles
}

fn push_deepseek_v3_attention_roles(roles: &mut Vec<WeightBlockRole>) {
    roles.extend([
        WeightBlockRole::DeepSeekQALoraProjection,
        WeightBlockRole::DeepSeekQALoraScaleInv,
        WeightBlockRole::DeepSeekQALoraNorm,
        WeightBlockRole::DeepSeekQBProjection,
        WeightBlockRole::DeepSeekQBScaleInv,
        WeightBlockRole::DeepSeekKvAProjection,
        WeightBlockRole::DeepSeekKvAScaleInv,
        WeightBlockRole::DeepSeekKvANorm,
        WeightBlockRole::DeepSeekKvBProjection,
        WeightBlockRole::DeepSeekKvBScaleInv,
        WeightBlockRole::OutputProjection,
        WeightBlockRole::DeepSeekOutputScaleInv,
    ]);
}

fn push_deepseek_v32_indexer_roles(roles: &mut Vec<WeightBlockRole>) {
    roles.extend([
        WeightBlockRole::DeepSeekIndexerQueryProjection,
        WeightBlockRole::DeepSeekIndexerQueryScaleInv,
        WeightBlockRole::DeepSeekIndexerKeyProjection,
        WeightBlockRole::DeepSeekIndexerKeyScaleInv,
        WeightBlockRole::DeepSeekIndexerKeyNorm,
        WeightBlockRole::DeepSeekIndexerKeyNormBias,
        WeightBlockRole::DeepSeekIndexerWeightsProjection,
    ]);
}

fn push_deepseek_v4_attention_roles(roles: &mut Vec<WeightBlockRole>) {
    roles.extend([
        WeightBlockRole::DeepSeekV4HcAttnBase,
        WeightBlockRole::DeepSeekV4HcAttnFn,
        WeightBlockRole::DeepSeekV4HcAttnScale,
        WeightBlockRole::DeepSeekV4HcFfnBase,
        WeightBlockRole::DeepSeekV4HcFfnFn,
        WeightBlockRole::DeepSeekV4HcFfnScale,
        WeightBlockRole::DeepSeekV4AttentionSink,
        WeightBlockRole::DeepSeekV4WqAProjection,
        WeightBlockRole::DeepSeekV4WqAScale,
        WeightBlockRole::DeepSeekV4WqBProjection,
        WeightBlockRole::DeepSeekV4WqBScale,
        WeightBlockRole::DeepSeekV4QNorm,
        WeightBlockRole::DeepSeekV4WkvProjection,
        WeightBlockRole::DeepSeekV4WkvScale,
        WeightBlockRole::DeepSeekV4KvNorm,
        WeightBlockRole::DeepSeekV4WoAProjection,
        WeightBlockRole::DeepSeekV4WoAScale,
        WeightBlockRole::DeepSeekV4WoBProjection,
        WeightBlockRole::DeepSeekV4WoBScale,
    ]);
}

fn push_deepseek_v4_compressor_roles(roles: &mut Vec<WeightBlockRole>, indexer: bool) {
    if indexer {
        roles.extend([
            WeightBlockRole::DeepSeekV4IndexerCompressorApe,
            WeightBlockRole::DeepSeekV4IndexerCompressorWkvProjection,
            WeightBlockRole::DeepSeekV4IndexerCompressorWgateProjection,
            WeightBlockRole::DeepSeekV4IndexerCompressorNorm,
        ]);
    } else {
        roles.extend([
            WeightBlockRole::DeepSeekV4CompressorApe,
            WeightBlockRole::DeepSeekV4CompressorWkvProjection,
            WeightBlockRole::DeepSeekV4CompressorWgateProjection,
            WeightBlockRole::DeepSeekV4CompressorNorm,
        ]);
    }
}

fn push_deepseek_v4_sparse_indexer_roles(roles: &mut Vec<WeightBlockRole>) {
    roles.extend([
        WeightBlockRole::DeepSeekV4IndexerWqBProjection,
        WeightBlockRole::DeepSeekV4IndexerWqBScale,
        WeightBlockRole::DeepSeekV4IndexerWeightsProjection,
    ]);
}

fn push_deepseek_dense_mlp_roles(roles: &mut Vec<WeightBlockRole>) {
    roles.extend([
        WeightBlockRole::GateProjection,
        WeightBlockRole::GateScaleInv,
        WeightBlockRole::UpProjection,
        WeightBlockRole::UpScaleInv,
        WeightBlockRole::DownProjection,
        WeightBlockRole::DownScaleInv,
    ]);
}

fn push_deepseek_moe_roles(
    metadata: &HfModelMetadata,
    layer: usize,
    roles: &mut Vec<WeightBlockRole>,
) {
    roles.push(WeightBlockRole::RouterProjection);
    if metadata.architecture == HfArchitectureKind::DeepSeekV4 {
        if layer < metadata.num_hash_layers.unwrap_or(0) {
            roles.push(WeightBlockRole::DeepSeekV4HashRouteTable);
        } else {
            roles.push(WeightBlockRole::RouterCorrectionBias);
        }
        push_deepseek_v4_moe_projection_roles(metadata, roles);
        return;
    }
    if metadata.topk_method.as_deref() == Some("noaux_tc") {
        roles.push(WeightBlockRole::RouterCorrectionBias);
    }
    push_deepseek_v3_moe_projection_roles(metadata, roles);
}

fn push_deepseek_v3_moe_projection_roles(
    metadata: &HfModelMetadata,
    roles: &mut Vec<WeightBlockRole>,
) {
    if metadata.shared_expert_intermediate_size.unwrap_or(0) > 0 {
        roles.extend([
            WeightBlockRole::SharedExpertGateProjection,
            WeightBlockRole::SharedExpertGateScaleInv,
            WeightBlockRole::SharedExpertUpProjection,
            WeightBlockRole::SharedExpertUpScaleInv,
            WeightBlockRole::SharedExpertDownProjection,
            WeightBlockRole::SharedExpertDownScaleInv,
        ]);
    }
    roles.extend([
        WeightBlockRole::ExpertGateProjection,
        WeightBlockRole::ExpertGateScaleInv,
        WeightBlockRole::ExpertUpProjection,
        WeightBlockRole::ExpertUpScaleInv,
        WeightBlockRole::ExpertDownProjection,
        WeightBlockRole::ExpertDownScaleInv,
    ]);
}

fn push_deepseek_v4_moe_projection_roles(
    metadata: &HfModelMetadata,
    roles: &mut Vec<WeightBlockRole>,
) {
    if metadata.shared_expert_intermediate_size.unwrap_or(0) > 0 {
        roles.extend([
            WeightBlockRole::SharedExpertGateProjection,
            WeightBlockRole::DeepSeekV4SharedExpertGateScale,
            WeightBlockRole::SharedExpertUpProjection,
            WeightBlockRole::DeepSeekV4SharedExpertUpScale,
            WeightBlockRole::SharedExpertDownProjection,
            WeightBlockRole::DeepSeekV4SharedExpertDownScale,
        ]);
    }
    roles.extend([
        WeightBlockRole::ExpertGateProjection,
        WeightBlockRole::DeepSeekV4ExpertGateScale,
        WeightBlockRole::ExpertUpProjection,
        WeightBlockRole::DeepSeekV4ExpertUpScale,
        WeightBlockRole::ExpertDownProjection,
        WeightBlockRole::DeepSeekV4ExpertDownScale,
    ]);
}

fn deepseek_runtime_cache_dtype(architecture: HfArchitectureKind) -> &'static str {
    match architecture {
        HfArchitectureKind::DeepSeekV3 => "bfloat16",
        HfArchitectureKind::DeepSeekV32 | HfArchitectureKind::DeepSeekV4 => "fp8_ds_mla",
        _ => "auto",
    }
}

fn require_kv_group(plan: &crate::hf::deepseek::DeepSeekVllmKvCachePlan, name: &str) -> Result<()> {
    if plan.groups.iter().any(|group| group.name == name) {
        Ok(())
    } else {
        Err(NervaError::InvalidArgument {
            reason: format!(
                "DeepSeek layer execution plan requires KV cache group {name}, but vLLM KV cache planner did not produce it"
            ),
        })
    }
}

pub fn deepseek_layer_report(metadata: &HfModelMetadata) -> DeepSeekLayerReport {
    let moe_layers = metadata
        .mlp_layer_types
        .iter()
        .filter(|kind| **kind == HfMlpLayerKind::SparseMoe)
        .count();
    let dense_mlp_layers = metadata.mlp_layer_types.len().saturating_sub(moe_layers);
    let mut v4_swa_layers = 0usize;
    let mut v4_c4_layers = 0usize;
    let mut v4_c128_layers = 0usize;
    let mut v4_indexer_layers = 0usize;
    if metadata.architecture == HfArchitectureKind::DeepSeekV4 {
        for layer in 0..metadata.num_hidden_layers {
            match metadata.compress_ratios.get(layer).copied().unwrap_or(0) {
                4 => {
                    v4_c4_layers += 1;
                    v4_indexer_layers += 1;
                }
                128 => v4_c128_layers += 1,
                _ => v4_swa_layers += 1,
            }
        }
    }
    DeepSeekLayerReport {
        moe_layers,
        dense_mlp_layers,
        v4_swa_layers,
        v4_c4_layers,
        v4_c128_layers,
        v4_indexer_layers,
        v4_hash_router_layers: metadata.num_hash_layers.unwrap_or(0),
    }
}

pub fn deepseek_required_execution_units(metadata: &HfModelMetadata) -> Vec<String> {
    match metadata.architecture {
        HfArchitectureKind::DeepSeekV3 => vec![
            "deepseek_v3_mla_prefill_decode".to_string(),
            "deepseek_v3_block_fp8_projection_gemm".to_string(),
            "deepseek_v3_grouped_moe_router_noaux_tc".to_string(),
            "deepseek_v3_split_fp8_expert_moe".to_string(),
            "deepseek_v3_mtp_optional".to_string(),
        ],
        HfArchitectureKind::DeepSeekV32 => vec![
            "deepseek_v3_mla_prefill_decode".to_string(),
            "deepseek_v3_block_fp8_projection_gemm".to_string(),
            "deepseek_v32_sparse_attention_indexer".to_string(),
            "deepseek_v3_grouped_moe_router_noaux_tc".to_string(),
            "deepseek_v3_split_fp8_expert_moe".to_string(),
        ],
        HfArchitectureKind::DeepSeekV4 => vec![
            "deepseek_v4_mhc_pre_post_head".to_string(),
            "deepseek_v4_mla_swa_cache".to_string(),
            "deepseek_v4_fp8_ds_mla_cache".to_string(),
            "deepseek_v4_c4_c128_compressor".to_string(),
            "deepseek_v4_sparse_indexer".to_string(),
            "deepseek_v4_parallel_attention_gemm_streams".to_string(),
            "deepseek_v4_hash_and_bias_router".to_string(),
            "deepseek_v4_megamoe_int8_fp4_experts".to_string(),
        ],
        _ => Vec::new(),
    }
}

pub fn deepseek_implemented_primitives(metadata: &HfModelMetadata) -> Vec<String> {
    if !metadata.architecture.is_deepseek() {
        return Vec::new();
    }

    let mut primitives = vec![
        "deepseek_weight_manifest_v3_v32_v4".to_string(),
        "fp8_e4m3fn_decode_matches_torch".to_string(),
        "e8m0_scale_upcast_matches_vllm_raw_exponent_path".to_string(),
        "fp8_e4m3fn_e8m0_block_dequant_reference".to_string(),
        "cuda_fp8_e4m3fn_e8m0_dequant_api".to_string(),
        "cuda_fp8_e4m3fn_e8m0_block_dequant_smoke".to_string(),
        "deepseek_vllm_kv_cache_spec_planner".to_string(),
        "deepseek_mla_decode_mqa_reference".to_string(),
        "cuda_deepseek_mla_decode_api".to_string(),
        "cuda_deepseek_mla_decode_mqa_smoke".to_string(),
        "deepseek_routed_moe_reference".to_string(),
        "cuda_deepseek_routed_moe_api".to_string(),
        "cuda_deepseek_routed_moe_smoke".to_string(),
        "deepseek_v3_grouped_sigmoid_router_reference".to_string(),
        "precision_moe_deepseek_v3_grouped_router".to_string(),
        "precision_moe_deepseek_router_correction_bias_load".to_string(),
        "cuda_deepseek_router_route_api".to_string(),
        "cuda_deepseek_v3_grouped_sigmoid_router_smoke".to_string(),
        "cuda_hf_sequence_deepseek_v3_grouped_router_runtime".to_string(),
        "cuda_hf_sequence_deepseek_descriptor_abi".to_string(),
        "cuda_hf_sequence_deepseek_footprint_accounting".to_string(),
        "cuda_hf_sequence_deepseek_native_layout_pack".to_string(),
        "cuda_hf_sequence_deepseek_execution_guard".to_string(),
    ];

    if matches!(
        metadata.architecture,
        HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32
    ) {
        primitives.push("cuda_hf_sequence_deepseek_v3_mla_kv_page_contents".to_string());
    }
    if metadata.architecture == HfArchitectureKind::DeepSeekV4 {
        primitives.push("deepseek_v4_mhc_compressor_indexer_manifest".to_string());
        primitives.push("deepseek_v4_hash_router_manifest".to_string());
        primitives.push("mxfp4_e2m1_e8m0_block_dequant_reference".to_string());
        primitives.push("cuda_mxfp4_e2m1_e8m0_dequant_api".to_string());
        primitives.push("cuda_mxfp4_e2m1_e8m0_block_dequant_smoke".to_string());
        primitives.push("deepseek_v4_sqrtsoftplus_hash_router_reference".to_string());
        primitives.push("precision_moe_deepseek_v4_sqrtsoftplus_router".to_string());
        primitives.push("deepseek_v4_hash_route_table_i64_loader".to_string());
        primitives.push("precision_moe_deepseek_v4_hash_route_table".to_string());
        primitives.push("cuda_deepseek_v4_sqrtsoftplus_hash_router_smoke".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_bias_router_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_hash_router_runtime".to_string());
        primitives.push("cuda_deepseek_qkv_rmsnorm_api".to_string());
        primitives.push("cuda_deepseek_qkv_rmsnorm_smoke".to_string());
        primitives.push("cuda_deepseek_fused_inv_rope_fp8_quant_api".to_string());
        primitives.push("cuda_deepseek_fused_inv_rope_fp8_quant_smoke".to_string());
        primitives.push("cuda_deepseek_fp8_ds_mla_kv_pack_api".to_string());
        primitives.push("cuda_deepseek_fp8_ds_mla_kv_pack_smoke".to_string());
    }
    if matches!(
        metadata.architecture,
        HfArchitectureKind::DeepSeekV32 | HfArchitectureKind::DeepSeekV4
    ) {
        primitives.push("cuda_deepseek_compressed_slot_mapping_api".to_string());
        primitives.push("cuda_deepseek_compressed_slot_mapping_smoke".to_string());
    }
    if metadata.architecture == HfArchitectureKind::DeepSeekV4 {
        primitives.push("cuda_deepseek_c128_topk_metadata_api".to_string());
        primitives.push("cuda_deepseek_c128_topk_metadata_smoke".to_string());
        primitives.push("cuda_deepseek_c4_indexer_topk_api".to_string());
        primitives.push("cuda_deepseek_c4_indexer_topk_smoke".to_string());
        primitives.push("cuda_deepseek_save_partial_states_api".to_string());
        primitives.push("cuda_deepseek_save_partial_states_smoke".to_string());
        primitives.push("cuda_deepseek_compress_norm_rope_fp8_cache_api".to_string());
        primitives.push("cuda_deepseek_compress_norm_rope_fp8_cache_smoke".to_string());
        primitives.push("cuda_deepseek_compress_norm_rope_mxfp4_cache_api".to_string());
        primitives.push("cuda_deepseek_compress_norm_rope_mxfp4_cache_smoke".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_index_topk_descriptor".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_compressed_scan_metrics".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_swa_window_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_page_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_page_contents".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_fp8_ds_mla_page_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_fp8_ds_mla_page_contents".to_string());
        primitives
            .push("cuda_hf_sequence_deepseek_v4_fp8_ds_mla_fullsize_page_contents".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_c128_fp8_ds_mla_page_contents".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_c4_sparse_topk_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_c4_topk_cover_all_shortcut".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_attention_aux_stream_resources".to_string());
    }

    primitives
}

pub fn deepseek_execution_unit_coverage(
    metadata: &HfModelMetadata,
) -> Vec<DeepSeekExecutionUnitCoverage> {
    deepseek_required_execution_units(metadata)
        .into_iter()
        .map(|unit| coverage_for_unit(metadata.architecture, unit))
        .collect()
}

pub fn deepseek_runtime_blocking_units(
    metadata: &HfModelMetadata,
) -> Vec<DeepSeekExecutionUnitCoverage> {
    deepseek_execution_unit_coverage(metadata)
        .into_iter()
        .filter(|unit| unit.status != "complete" && unit.status != "optional_missing")
        .collect()
}

pub fn deepseek_exact_runtime_blocked_reason(metadata: &HfModelMetadata) -> String {
    let blockers = deepseek_runtime_blocking_units(metadata);
    let blocker_summary = if blockers.is_empty() {
        "no blocking execution units are recorded".to_string()
    } else {
        blockers
            .iter()
            .map(|unit| {
                format!(
                    "{} is {}; remaining gaps: {}",
                    unit.unit,
                    unit.status,
                    unit.remaining_gaps.join("; ")
                )
            })
            .collect::<Vec<_>>()
            .join(" | ")
    };

    format!(
        "HF architecture {} is recognized and its DeepSeek weight/KV layout is planned, but the exact runtime has not completed DeepSeek MLA attention, block-quantized FP8/FP4 projection GEMM, compressed KV/indexer state, and DeepSeek grouped MoE routing. Blocking execution units: {blocker_summary}",
        metadata.architecture.as_str(),
    )
}

fn coverage_for_unit(
    architecture: HfArchitectureKind,
    unit: String,
) -> DeepSeekExecutionUnitCoverage {
    let (status, validated_primitives, remaining_gaps): (&'static str, &[&str], &[&str]) = match (
        architecture,
        unit.as_str(),
    ) {
        (
            HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32,
            "deepseek_v3_mla_prefill_decode",
        ) => (
            "partial",
            &[
                "deepseek_vllm_kv_cache_spec_planner",
                "deepseek_mla_decode_mqa_reference",
                "cuda_deepseek_mla_decode_api",
                "cuda_deepseek_mla_decode_mqa_smoke",
                "cuda_hf_sequence_deepseek_descriptor_abi",
                "cuda_hf_sequence_deepseek_footprint_accounting",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_execution_guard",
                "cuda_hf_sequence_deepseek_v3_mla_kv_page_contents",
            ],
            &[
                "integrate MLA prefill/decode into exact runtime",
                "consume DeepSeek native sequence layout offsets in MLA kernels",
                "run full-size V3 MLA KV page differential against vLLM",
                "match vLLM DeepseekV2MLAAttention output numerics",
            ],
        ),
        (
            HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32,
            "deepseek_v3_block_fp8_projection_gemm",
        ) => (
            "partial",
            &[
                "fp8_e4m3fn_decode_matches_torch",
                "e8m0_scale_upcast_matches_vllm_raw_exponent_path",
                "fp8_e4m3fn_e8m0_block_dequant_reference",
                "cuda_fp8_e4m3fn_e8m0_dequant_api",
                "cuda_fp8_e4m3fn_e8m0_block_dequant_smoke",
                "cuda_hf_sequence_deepseek_footprint_accounting",
                "cuda_hf_sequence_deepseek_native_layout_pack",
            ],
            &[
                "fuse block-FP8 dequant with projection GEMM",
                "consume packed DeepSeek q_a/kv_a/q_b/kv_b/o projection scale offsets in decode",
                "benchmark projection throughput against vLLM fused kernels",
            ],
        ),
        (HfArchitectureKind::DeepSeekV32, "deepseek_v32_sparse_attention_indexer") => (
            "partial",
            &[
                "deepseek_vllm_kv_cache_spec_planner",
                "cuda_deepseek_compressed_slot_mapping_api",
                "cuda_deepseek_compressed_slot_mapping_smoke",
                "cuda_hf_sequence_deepseek_native_layout_pack",
            ],
            &[
                "consume packed V3.2 sparse indexer query/key/weights offsets in runtime",
                "store vLLM-compatible indexer cache pages",
                "verify selected sparse blocks against vLLM",
            ],
        ),
        (
            HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32,
            "deepseek_v3_grouped_moe_router_noaux_tc",
        ) => (
            "partial",
            &[
                "deepseek_v3_grouped_sigmoid_router_reference",
                "precision_moe_deepseek_v3_grouped_router",
                "precision_moe_deepseek_router_correction_bias_load",
                "cuda_deepseek_router_route_api",
                "cuda_deepseek_v3_grouped_sigmoid_router_smoke",
                "cuda_hf_sequence_deepseek_v3_grouped_router_runtime",
                "cuda_hf_sequence_deepseek_native_layout_pack",
            ],
            &["verify full-layer routed outputs against vLLM"],
        ),
        (
            HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32,
            "deepseek_v3_split_fp8_expert_moe",
        ) => (
            "partial",
            &[
                "deepseek_routed_moe_reference",
                "cuda_deepseek_routed_moe_api",
                "cuda_deepseek_routed_moe_smoke",
                "fp8_e4m3fn_e8m0_block_dequant_reference",
                "cuda_fp8_e4m3fn_e8m0_dequant_api",
                "cuda_fp8_e4m3fn_e8m0_block_dequant_smoke",
                "cuda_hf_sequence_deepseek_native_layout_pack",
            ],
            &[
                "run routed expert gate/up/down GEMMs with checkpoint FP8 scales",
                "integrate shared experts and routed output accumulation",
                "benchmark fused MoE against vLLM FusedMoE",
            ],
        ),
        (HfArchitectureKind::DeepSeekV3, "deepseek_v3_mtp_optional") => (
            "optional_missing",
            &[],
            &["optional MTP draft layers are not implemented"],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_mhc_pre_post_head") => (
            "partial",
            &[
                "deepseek_v4_mhc_compressor_indexer_manifest",
                "cuda_deepseek_qkv_rmsnorm_api",
                "cuda_deepseek_qkv_rmsnorm_smoke",
                "cuda_deepseek_fused_inv_rope_fp8_quant_api",
                "cuda_deepseek_fused_inv_rope_fp8_quant_smoke",
                "cuda_hf_sequence_deepseek_descriptor_abi",
                "cuda_hf_sequence_deepseek_footprint_accounting",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_execution_guard",
            ],
            &[
                "integrate fused Q/KV RMSNorm into MHC pre-head runtime",
                "integrate fused inverse RoPE FP8 quant into O projection runtime",
                "implement remaining MHC pre/post-head transforms",
                "verify MHC head/attention/FFN scale handling against vLLM",
            ],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_mla_swa_cache") => (
            "partial",
            &[
                "deepseek_vllm_kv_cache_spec_planner",
                "deepseek_mla_decode_mqa_reference",
                "cuda_deepseek_mla_decode_api",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_v4_swa_window_runtime",
                "cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_page_runtime",
                "cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_page_contents",
                "cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_nonzero_page_contents",
                "cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_fullsize_page_contents",
            ],
            &[
                "run full-size V4 SWA fp8_ds_mla page differential against vLLM FlashMLA",
                "replace serial SWA page reader with the vLLM FlashMLA/FlashInfer kernel path",
            ],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_fp8_ds_mla_cache") => (
            "partial",
            &[
                "deepseek_vllm_kv_cache_spec_planner",
                "deepseek_mla_decode_mqa_reference",
                "cuda_deepseek_mla_decode_api",
                "fp8_e4m3fn_e8m0_block_dequant_reference",
                "cuda_fp8_e4m3fn_e8m0_dequant_api",
                "cuda_fp8_e4m3fn_e8m0_block_dequant_smoke",
                "cuda_deepseek_fp8_ds_mla_kv_pack_api",
                "cuda_deepseek_fp8_ds_mla_kv_pack_smoke",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_v4_fp8_ds_mla_page_runtime",
                "cuda_hf_sequence_deepseek_v4_fp8_ds_mla_page_contents",
                "cuda_hf_sequence_deepseek_v4_fp8_ds_mla_fullsize_page_contents",
                "cuda_hf_sequence_deepseek_v4_c128_fp8_ds_mla_page_contents",
            ],
            &["run full-size V4 compressed fp8_ds_mla page differential against vLLM FlashMLA"],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_c4_c128_compressor") => (
            "partial",
            &[
                "deepseek_v4_mhc_compressor_indexer_manifest",
                "cuda_deepseek_qkv_rmsnorm_api",
                "cuda_deepseek_qkv_rmsnorm_smoke",
                "cuda_deepseek_save_partial_states_api",
                "cuda_deepseek_save_partial_states_smoke",
                "cuda_deepseek_compress_norm_rope_fp8_cache_api",
                "cuda_deepseek_compress_norm_rope_fp8_cache_smoke",
                "cuda_deepseek_compress_norm_rope_mxfp4_cache_api",
                "cuda_deepseek_compress_norm_rope_mxfp4_cache_smoke",
                "cuda_deepseek_compressed_slot_mapping_api",
                "cuda_deepseek_compressed_slot_mapping_smoke",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_v4_fp8_ds_mla_fullsize_page_contents",
                "cuda_hf_sequence_deepseek_v4_c128_fp8_ds_mla_page_contents",
            ],
            &[
                "replace serial C4/C128 compressor path with the vLLM fused compressor kernel pattern",
                "run C4/C128 compressor cache differential against vLLM",
            ],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_sparse_indexer") => (
            "partial",
            &[
                "deepseek_vllm_kv_cache_spec_planner",
                "deepseek_v4_mhc_compressor_indexer_manifest",
                "cuda_deepseek_compressed_slot_mapping_api",
                "cuda_deepseek_compressed_slot_mapping_smoke",
                "cuda_deepseek_c128_topk_metadata_api",
                "cuda_deepseek_c128_topk_metadata_smoke",
                "cuda_deepseek_c4_indexer_topk_api",
                "cuda_deepseek_c4_indexer_topk_smoke",
                "cuda_hf_sequence_deepseek_v4_index_topk_descriptor",
                "cuda_hf_sequence_deepseek_v4_compressed_scan_metrics",
                "cuda_hf_sequence_deepseek_v4_c4_sparse_topk_runtime",
                "cuda_hf_sequence_deepseek_v4_c4_topk_cover_all_shortcut",
                "cuda_hf_sequence_deepseek_native_layout_pack",
            ],
            &[
                "verify C4 sparse top-k numerics against vLLM end-to-end",
                "integrate split paged-logits/top-k sparse scorer into full decode runtime for partial-cover cases",
            ],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_parallel_attention_gemm_streams") => (
            "partial",
            &["cuda_hf_sequence_deepseek_v4_attention_aux_stream_resources"],
            &[
                "schedule attention GEMM/compressor/indexer kernels onto the V4 aux streams like vLLM",
                "measure stream overlap against vLLM DeepseekV4 attention",
            ],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_hash_and_bias_router") => (
            "partial",
            &[
                "deepseek_v4_hash_router_manifest",
                "deepseek_v4_sqrtsoftplus_hash_router_reference",
                "precision_moe_deepseek_v4_sqrtsoftplus_router",
                "deepseek_v4_hash_route_table_i64_loader",
                "precision_moe_deepseek_v4_hash_route_table",
                "cuda_deepseek_router_route_api",
                "cuda_deepseek_v4_sqrtsoftplus_hash_router_smoke",
                "cuda_hf_sequence_deepseek_v4_bias_router_runtime",
                "cuda_hf_sequence_deepseek_v4_hash_router_runtime",
                "cuda_hf_sequence_deepseek_native_layout_pack",
            ],
            &["verify full-layer routed outputs against vLLM"],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_megamoe_int8_fp4_experts") => (
            "partial",
            &[
                "mxfp4_e2m1_e8m0_block_dequant_reference",
                "cuda_mxfp4_e2m1_e8m0_dequant_api",
                "cuda_mxfp4_e2m1_e8m0_block_dequant_smoke",
                "deepseek_routed_moe_reference",
                "cuda_deepseek_routed_moe_api",
                "cuda_deepseek_routed_moe_smoke",
                "cuda_hf_sequence_deepseek_native_layout_pack",
            ],
            &[
                "implement V4 MegaMoE int8/fp4 expert kernels",
                "support expert-parallel physical/logical expert mapping",
                "benchmark MegaMoE against vLLM deep_gemm_mega_moe/FusedMoE",
            ],
        ),
        _ => (
            "missing",
            &[],
            &["no DeepSeek coverage mapping exists for this unit"],
        ),
    };

    DeepSeekExecutionUnitCoverage {
        unit,
        status,
        validated_primitives: validated_primitives
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        remaining_gaps: remaining_gaps
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
    }
}
