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
    pub compressor_state_kv_cache_group: Option<String>,
    pub indexer_compressor_state_kv_cache_group: Option<String>,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MhcWarmupTokenPlan {
    pub tokens: usize,
    pub mhc_pre_num_split: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MhcWarmupPlan {
    pub max_tokens: usize,
    pub hidden_size: usize,
    pub hc_mult: usize,
    pub num_sms: usize,
    pub token_sizes: Vec<DeepSeekV4MhcWarmupTokenPlan>,
}

pub const DEEPSEEK_V4_MHC_AUTO_WARMUP_MAX_TOKENS: usize = 16_384;
pub const DEEPSEEK_V4_MHC_DEFAULT_TOKEN_SIZE_CANDIDATES: [usize; 15] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16_384,
];

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
                    None,
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
                let uses_sparse_indexer = deepseek_v32_layer_uses_sparse_indexer(metadata, layer)?;
                if uses_sparse_indexer {
                    require_kv_group(&kv_plan, "v3_2_sparse_indexer")?;
                }
                Ok(layer_execution(
                    metadata,
                    layer,
                    DeepSeekAttentionExecutionKind::V32MlaWithIndexer,
                    1,
                    "v3_2_main_mla",
                    uses_sparse_indexer.then_some("v3_2_sparse_indexer"),
                    None,
                    None,
                    false,
                    uses_sparse_indexer,
                    uses_sparse_indexer,
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
                            None,
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
                        require_kv_group(&kv_plan, "v4_c4_compressor_state")?;
                        require_kv_group(&kv_plan, "v4_c4_indexer_compressor_state")?;
                        Ok(layer_execution(
                            metadata,
                            layer,
                            DeepSeekAttentionExecutionKind::V4CompressedMlaWithSparseIndexer,
                            compress_ratio,
                            "v4_c4_mla",
                            Some("v4_c4_mla_indexer"),
                            Some("v4_c4_compressor_state"),
                            Some("v4_c4_indexer_compressor_state"),
                            false,
                            true,
                            true,
                            true,
                        ))
                    }
                    128 => {
                        require_kv_group(&kv_plan, "v4_c128_mla")?;
                        require_kv_group(&kv_plan, "v4_c128_compressor_state")?;
                        Ok(layer_execution(
                            metadata,
                            layer,
                            DeepSeekAttentionExecutionKind::V4CompressedMla,
                            compress_ratio,
                            "v4_c128_mla",
                            None,
                            Some("v4_c128_compressor_state"),
                            None,
                            false,
                            false,
                            false,
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

fn deepseek_v32_layer_uses_sparse_indexer(
    metadata: &HfModelMetadata,
    layer: usize,
) -> Result<bool> {
    if metadata.index_topk.unwrap_or(0) == 0 {
        return Ok(false);
    }
    if let Some(pattern) = metadata.index_topk_pattern.get(layer) {
        return Ok(pattern != "S");
    }
    let freq = metadata.index_topk_freq.unwrap_or(1);
    if freq == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek V3.2 index_topk_freq must be non-zero".to_string(),
        });
    }
    let offset = metadata.index_skip_topk_offset.unwrap_or(2);
    let schedule_pos = layer.saturating_add(1).saturating_sub(offset);
    Ok(schedule_pos % freq == 0)
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

pub fn deepseek_v4_mhc_warmup_token_sizes(
    max_tokens: usize,
    cudagraph_capture_sizes: &[usize],
) -> Vec<usize> {
    if max_tokens == 0 {
        return Vec::new();
    }

    let max_auto_tokens = max_tokens.min(DEEPSEEK_V4_MHC_AUTO_WARMUP_MAX_TOKENS);
    let mut candidates = Vec::with_capacity(
        DEEPSEEK_V4_MHC_DEFAULT_TOKEN_SIZE_CANDIDATES.len() + cudagraph_capture_sizes.len() + 1,
    );
    candidates.extend(DEEPSEEK_V4_MHC_DEFAULT_TOKEN_SIZE_CANDIDATES);
    candidates.extend(cudagraph_capture_sizes.iter().copied());
    candidates.push(max_auto_tokens);
    candidates.retain(|size| (1..=max_auto_tokens).contains(size));
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

pub fn deepseek_v4_mhc_pre_num_split(
    num_tokens: usize,
    hidden_size: usize,
    hc_mult: usize,
    num_sms: usize,
) -> Result<usize> {
    if num_tokens == 0 || hidden_size == 0 || hc_mult == 0 || num_sms == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "DeepSeek V4 mHC split planning requires non-zero tokens, hidden size, hc_mult, and SM count".to_string(),
        });
    }

    let k = hidden_size
        .checked_mul(hc_mult)
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: format!(
                "DeepSeek V4 mHC split planning overflowed: hidden_size {hidden_size} * hc_mult {hc_mult}"
            ),
        })?;
    let grid_size = ceil_div(num_tokens, 64);
    let split_k = num_sms / grid_size;
    let num_block_k = ceil_div(k, 64);
    Ok(split_k.min(num_block_k / 4).max(1))
}

pub fn plan_deepseek_v4_mhc_warmup(
    metadata: &HfModelMetadata,
    max_tokens: usize,
    cudagraph_capture_sizes: &[usize],
    num_sms: usize,
) -> Result<DeepSeekV4MhcWarmupPlan> {
    if metadata.architecture != HfArchitectureKind::DeepSeekV4 {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "DeepSeek V4 mHC warmup planning requires DeepSeek V4, got {}",
                metadata.architecture.as_str()
            ),
        });
    }
    let hc_mult = metadata
        .hc_mult
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "DeepSeek V4 mHC warmup planning requires hc_mult metadata".to_string(),
        })?;

    let token_sizes = deepseek_v4_mhc_warmup_token_sizes(max_tokens, cudagraph_capture_sizes)
        .into_iter()
        .map(|tokens| {
            Ok(DeepSeekV4MhcWarmupTokenPlan {
                tokens,
                mhc_pre_num_split: deepseek_v4_mhc_pre_num_split(
                    tokens,
                    metadata.hidden_size,
                    hc_mult,
                    num_sms,
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(DeepSeekV4MhcWarmupPlan {
        max_tokens,
        hidden_size: metadata.hidden_size,
        hc_mult,
        num_sms,
        token_sizes,
    })
}

fn layer_execution(
    metadata: &HfModelMetadata,
    layer: usize,
    attention_kind: DeepSeekAttentionExecutionKind,
    compress_ratio: usize,
    primary_kv_cache_group: &str,
    indexer_kv_cache_group: Option<&str>,
    compressor_state_kv_cache_group: Option<&str>,
    indexer_compressor_state_kv_cache_group: Option<&str>,
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
        compressor_state_kv_cache_group: compressor_state_kv_cache_group.map(str::to_string),
        indexer_compressor_state_kv_cache_group: indexer_compressor_state_kv_cache_group
            .map(str::to_string),
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
                push_deepseek_v4_compressor_roles(
                    &mut roles,
                    false,
                    metadata.expert_dtype.as_deref() == Some("bf16"),
                );
            }
            if execution.uses_sparse_indexer {
                push_deepseek_v4_sparse_indexer_roles(
                    &mut roles,
                    metadata.expert_dtype.as_deref() == Some("bf16"),
                );
                push_deepseek_v4_compressor_roles(
                    &mut roles,
                    true,
                    metadata.expert_dtype.as_deref() == Some("bf16"),
                );
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

fn push_deepseek_v4_compressor_roles(
    roles: &mut Vec<WeightBlockRole>,
    indexer: bool,
    quantized_projection: bool,
) {
    if indexer {
        roles.extend([
            WeightBlockRole::DeepSeekV4IndexerCompressorApe,
            WeightBlockRole::DeepSeekV4IndexerCompressorWkvProjection,
            WeightBlockRole::DeepSeekV4IndexerCompressorWgateProjection,
            WeightBlockRole::DeepSeekV4IndexerCompressorNorm,
        ]);
        if quantized_projection {
            roles.extend([
                WeightBlockRole::DeepSeekV4IndexerCompressorWkvScale,
                WeightBlockRole::DeepSeekV4IndexerCompressorWgateScale,
            ]);
        }
    } else {
        roles.extend([
            WeightBlockRole::DeepSeekV4CompressorApe,
            WeightBlockRole::DeepSeekV4CompressorWkvProjection,
            WeightBlockRole::DeepSeekV4CompressorWgateProjection,
            WeightBlockRole::DeepSeekV4CompressorNorm,
        ]);
        if quantized_projection {
            roles.extend([
                WeightBlockRole::DeepSeekV4CompressorWkvScale,
                WeightBlockRole::DeepSeekV4CompressorWgateScale,
            ]);
        }
    }
}

fn push_deepseek_v4_sparse_indexer_roles(
    roles: &mut Vec<WeightBlockRole>,
    quantized_projection: bool,
) {
    roles.extend([
        WeightBlockRole::DeepSeekV4IndexerWqBProjection,
        WeightBlockRole::DeepSeekV4IndexerWqBScale,
        WeightBlockRole::DeepSeekV4IndexerWeightsProjection,
    ]);
    if quantized_projection {
        roles.push(WeightBlockRole::DeepSeekV4IndexerWeightsScale);
    }
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

fn ceil_div(value: usize, divisor: usize) -> usize {
    debug_assert!(divisor != 0);
    value / divisor + usize::from(value % divisor != 0)
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
            "deepseek_v3_vllm_e2e_parity".to_string(),
            "deepseek_v3_mtp_optional".to_string(),
        ],
        HfArchitectureKind::DeepSeekV32 => vec![
            "deepseek_v3_mla_prefill_decode".to_string(),
            "deepseek_v3_block_fp8_projection_gemm".to_string(),
            "deepseek_v32_sparse_attention_indexer".to_string(),
            "deepseek_v3_grouped_moe_router_noaux_tc".to_string(),
            "deepseek_v3_split_fp8_expert_moe".to_string(),
            "deepseek_v32_vllm_e2e_parity".to_string(),
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
            "deepseek_v4_vllm_e2e_parity".to_string(),
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
        "cuda_fp8_e4m3fn_f32_scale_encoded_gemm_tokens_api".to_string(),
        "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_api".to_string(),
        "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_token4_weight_reuse".to_string(),
        "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_row8_token4_input_reuse".to_string(),
        "deepseek_vllm_kv_cache_spec_planner".to_string(),
        "deepseek_mla_decode_mqa_reference".to_string(),
        "deepseek_mla_prefill_causal_mqa_reference".to_string(),
        "cuda_deepseek_mla_decode_api".to_string(),
        "cuda_deepseek_mla_decode_mqa_smoke".to_string(),
        "deepseek_routed_moe_reference".to_string(),
        "deepseek_full_routed_moe_reference".to_string(),
        "cuda_deepseek_routed_moe_api".to_string(),
        "cuda_deepseek_routed_moe_smoke".to_string(),
        "deepseek_v3_grouped_sigmoid_router_reference".to_string(),
        "deepseek_v3_full_routed_moe_noaux_tc_reference".to_string(),
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
        primitives.push("cuda_hf_sequence_deepseek_v3_mla_fullsize_kv_page_contents".to_string());
        primitives.push(
            "cuda_hf_sequence_deepseek_v3_mla_batched_single_layer_prefill_cache_rows".to_string(),
        );
        primitives.push("cuda_hf_sequence_deepseek_v3_mla_parallel_head_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v3_split_fp8_moe_runtime".to_string());
    }
    if metadata.architecture == HfArchitectureKind::DeepSeekV4 {
        primitives.push("deepseek_v4_mhc_compressor_indexer_manifest".to_string());
        primitives.push("deepseek_v4_mhc_warmup_plan_matches_vllm".to_string());
        primitives.push("deepseek_v4_mhc_pre_post_head_reference_matches_vllm_torch".to_string());
        primitives.push("cuda_deepseek_mhc_pre_api".to_string());
        primitives.push("cuda_deepseek_mhc_pre_smoke".to_string());
        primitives.push("cuda_deepseek_mhc_post_api".to_string());
        primitives.push("cuda_deepseek_mhc_post_smoke".to_string());
        primitives.push("cuda_deepseek_mhc_fused_post_pre_api".to_string());
        primitives.push("cuda_deepseek_mhc_fused_post_pre_smoke".to_string());
        primitives.push("cuda_deepseek_mhc_head_api".to_string());
        primitives.push("cuda_deepseek_mhc_head_smoke".to_string());
        primitives.push("deepseek_v4_hash_router_manifest".to_string());
        primitives.push("mxfp4_e2m1_e8m0_block_dequant_reference".to_string());
        primitives.push("cuda_mxfp4_e2m1_e8m0_dequant_api".to_string());
        primitives.push("cuda_mxfp4_e2m1_e8m0_block_dequant_smoke".to_string());
        primitives.push("cuda_deepseek_megamoe_prepare_api".to_string());
        primitives.push("cuda_deepseek_megamoe_prepare_smoke".to_string());
        primitives.push("cuda_deepseek_megamoe_eplb_mapping_api".to_string());
        primitives.push("cuda_deepseek_megamoe_eplb_mapping_smoke".to_string());
        primitives.push("cuda_deepseek_megamoe_fp8_fp4_expert_api".to_string());
        primitives.push("cuda_deepseek_megamoe_fp8_fp4_expert_smoke".to_string());
        primitives.push("deepseek_v4_sqrtsoftplus_hash_router_reference".to_string());
        primitives.push("deepseek_v4_full_routed_moe_hash_reference".to_string());
        primitives.push("precision_moe_deepseek_v4_sqrtsoftplus_router".to_string());
        primitives.push("deepseek_v4_hash_route_table_i64_loader".to_string());
        primitives.push("precision_moe_deepseek_v4_hash_route_table".to_string());
        primitives.push("cuda_deepseek_v4_sqrtsoftplus_hash_router_smoke".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_bias_router_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_hash_router_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_sparse_moe_route_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_mxfp4_expert_gate_up_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_mxfp4_expert_down_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_parallel_sparse_moe_runtime".to_string());
        primitives.push("deepseek_qkv_rmsnorm_reference".to_string());
        primitives.push("cuda_deepseek_qkv_rmsnorm_api".to_string());
        primitives.push("cuda_deepseek_qkv_rmsnorm_smoke".to_string());
        primitives.push("cuda_deepseek_fused_inv_rope_fp8_quant_api".to_string());
        primitives.push("cuda_deepseek_fused_inv_rope_fp8_quant_smoke".to_string());
        primitives.push("cuda_deepseek_fp8_ds_mla_kv_pack_api".to_string());
        primitives.push("cuda_deepseek_fp8_ds_mla_kv_pack_smoke".to_string());
    }
    if metadata.architecture == HfArchitectureKind::DeepSeekV32 {
        primitives.push("deepseek_v32_vllm_index_topk_skip_schedule".to_string());
        primitives.push("cuda_hf_sequence_deepseek_packed_kv_footprint_accounting".to_string());
        primitives.push("cuda_deepseek_v32_fp8_ds_mla_kv_pack_token_row".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v32_fp8_ds_mla_page_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v32_fp8_ds_mla_page_contents".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v32_indexer_kv_page_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v32_indexer_kv_page_contents".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v32_indexer_query_state_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v32_indexer_query_state_contents".to_string());
        primitives.push(
            "cuda_hf_sequence_deepseek_v32_sparse_indexer_batched_single_layer_prefill_state"
                .to_string(),
        );
        primitives.push("cuda_hf_sequence_deepseek_v32_sparse_topk_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v32_sparse_topk_selection_hash".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v32_sparse_attention_consumes_topk".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v32_sparse_attention_output_hash".to_string());
        primitives.push(
            "cuda_hf_sequence_deepseek_v32_sparse_attention_topk2_output_differential".to_string(),
        );
        primitives.push("cuda_hf_sequence_deepseek_v32_sparse_mla_kv_b_scale_runtime".to_string());
        primitives.push(
            "cuda_hf_sequence_deepseek_v32_output_projection_scale_logits_runtime".to_string(),
        );
        primitives.push(
            "cuda_hf_sequence_deepseek_v32_q_a_kv_a_q_b_scale_sparse_decode_runtime".to_string(),
        );
    }
    if matches!(
        metadata.architecture,
        HfArchitectureKind::DeepSeekV32 | HfArchitectureKind::DeepSeekV4
    ) {
        primitives.push("deepseek_compressed_slot_mapping_reference".to_string());
        primitives.push("cuda_deepseek_compressed_slot_mapping_api".to_string());
        primitives.push("cuda_deepseek_compressed_slot_mapping_smoke".to_string());
    }
    if metadata.architecture == HfArchitectureKind::DeepSeekV4 {
        primitives.push("cuda_hf_sequence_deepseek_packed_kv_footprint_accounting".to_string());
        primitives.push("deepseek_c128_topk_metadata_reference".to_string());
        primitives.push("cuda_deepseek_c128_topk_metadata_api".to_string());
        primitives.push("cuda_deepseek_c128_topk_metadata_smoke".to_string());
        primitives.push("deepseek_c4_indexer_topk_reference".to_string());
        primitives.push("cuda_deepseek_c4_indexer_topk_api".to_string());
        primitives.push("cuda_deepseek_c4_indexer_topk_smoke".to_string());
        primitives.push("deepseek_save_partial_states_reference".to_string());
        primitives.push("cuda_deepseek_save_partial_states_api".to_string());
        primitives.push("cuda_deepseek_save_partial_states_smoke".to_string());
        primitives.push("deepseek_compress_norm_rope_fp8_cache_reference".to_string());
        primitives.push("cuda_deepseek_compress_norm_rope_fp8_cache_api".to_string());
        primitives.push("cuda_deepseek_compress_norm_rope_fp8_cache_smoke".to_string());
        primitives.push("cuda_deepseek_compress_norm_rope_mxfp4_cache_api".to_string());
        primitives.push("cuda_deepseek_compress_norm_rope_mxfp4_cache_smoke".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_index_topk_descriptor".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_compressed_scan_metrics".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_swa_window_runtime".to_string());
        primitives
            .push("cuda_hf_sequence_deepseek_v4_swa_parallel_head_attention_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_page_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_page_contents".to_string());
        primitives
            .push("cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_nonzero_page_contents".to_string());
        primitives
            .push("cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_fullsize_page_contents".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_fp8_ds_mla_page_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_fp8_ds_mla_page_contents".to_string());
        primitives
            .push("cuda_hf_sequence_deepseek_v4_fp8_ds_mla_fullsize_page_contents".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_c128_fp8_ds_mla_page_contents".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_c4_sparse_topk_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_c4_sparse_topk_selection_hash".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_sparse_attention_swa_plus_topk".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_c4_topk_cover_all_shortcut".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_attention_aux_stream_resources".to_string());
        primitives
            .push("cuda_hf_sequence_deepseek_v4_external_output_projection_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_mhc_sequence_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_mhc_head_final_norm_runtime".to_string());
        primitives.push("cuda_hf_sequence_deepseek_v4_mhc_native_profile_runtime".to_string());
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

pub fn validate_deepseek_exact_runtime_contract(metadata: &HfModelMetadata) -> Result<()> {
    if !metadata.architecture.is_deepseek() {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "DeepSeek runtime contract requires a DeepSeek architecture, got {}",
                metadata.architecture.as_str()
            ),
        });
    }
    let blockers = deepseek_runtime_blocking_units(metadata);
    if blockers.is_empty() {
        Ok(())
    } else {
        Err(NervaError::InvalidArgument {
            reason: deepseek_exact_runtime_blocked_reason_from_blockers(metadata, &blockers),
        })
    }
}

pub fn deepseek_exact_runtime_blocked_reason(metadata: &HfModelMetadata) -> String {
    let blockers = deepseek_runtime_blocking_units(metadata);
    deepseek_exact_runtime_blocked_reason_from_blockers(metadata, &blockers)
}

fn deepseek_exact_runtime_blocked_reason_from_blockers(
    metadata: &HfModelMetadata,
    blockers: &[DeepSeekExecutionUnitCoverage],
) -> String {
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
                "deepseek_mla_prefill_causal_mqa_reference",
                "cuda_deepseek_mla_decode_api",
                "cuda_deepseek_mla_decode_mqa_smoke",
                "cuda_hf_sequence_deepseek_descriptor_abi",
                "cuda_hf_sequence_deepseek_footprint_accounting",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_execution_guard",
                "cuda_hf_sequence_deepseek_v3_mla_kv_page_contents",
                "cuda_hf_sequence_deepseek_v3_mla_fullsize_kv_page_contents",
                "cuda_hf_sequence_deepseek_v3_mla_batched_single_layer_prefill_cache_rows",
                "cuda_hf_sequence_deepseek_v3_mla_parallel_head_runtime",
            ],
            &[
                "extend token-batched CUDA MLA prefill from single-layer cache population to full multi-layer vLLM-equivalent prefill",
                "consume DeepSeek native sequence layout offsets in MLA kernels",
                "run direct full-size V3 MLA KV page differential against vLLM runtime",
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
                "cuda_fp8_e4m3fn_f32_scale_encoded_gemm_tokens_api",
                "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_api",
                "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_token4_weight_reuse",
                "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_row8_token4_input_reuse",
                "cuda_hf_sequence_deepseek_footprint_accounting",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_v32_sparse_mla_kv_b_scale_runtime",
                "cuda_hf_sequence_deepseek_v32_output_projection_scale_logits_runtime",
                "cuda_hf_sequence_deepseek_v32_q_a_kv_a_q_b_scale_sparse_decode_runtime",
            ],
            &[
                "replace scalar row8/token4 FP8 projection tile with vLLM-class tensor-core/DeepGEMM kernel",
                "benchmark projection throughput against vLLM fused kernels",
            ],
        ),
        (HfArchitectureKind::DeepSeekV32, "deepseek_v32_sparse_attention_indexer") => (
            "partial",
            &[
                "deepseek_vllm_kv_cache_spec_planner",
                "deepseek_v32_vllm_index_topk_skip_schedule",
                "deepseek_compressed_slot_mapping_reference",
                "cuda_deepseek_compressed_slot_mapping_api",
                "cuda_deepseek_compressed_slot_mapping_smoke",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_packed_kv_footprint_accounting",
                "cuda_deepseek_v32_fp8_ds_mla_kv_pack_token_row",
                "cuda_hf_sequence_deepseek_v32_fp8_ds_mla_page_runtime",
                "cuda_hf_sequence_deepseek_v32_fp8_ds_mla_page_contents",
                "cuda_hf_sequence_deepseek_v32_indexer_kv_page_runtime",
                "cuda_hf_sequence_deepseek_v32_indexer_kv_page_contents",
                "cuda_hf_sequence_deepseek_v32_indexer_query_state_runtime",
                "cuda_hf_sequence_deepseek_v32_indexer_query_state_contents",
                "cuda_hf_sequence_deepseek_v32_sparse_indexer_batched_single_layer_prefill_state",
                "cuda_hf_sequence_deepseek_v32_sparse_topk_runtime",
                "cuda_hf_sequence_deepseek_v32_sparse_topk_selection_hash",
                "cuda_hf_sequence_deepseek_v32_sparse_attention_consumes_topk",
                "cuda_hf_sequence_deepseek_v32_sparse_attention_output_hash",
                "cuda_hf_sequence_deepseek_v32_sparse_attention_topk2_output_differential",
            ],
            &[
                "run full-size V3.2 sparse MLA attention differential against vLLM runtime",
                "benchmark V3.2 sparse MLA decode against vLLM FlashMLA sparse decode",
            ],
        ),
        (
            HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32,
            "deepseek_v3_grouped_moe_router_noaux_tc",
        ) => (
            "partial",
            &[
                "deepseek_v3_grouped_sigmoid_router_reference",
                "deepseek_v3_full_routed_moe_noaux_tc_reference",
                "precision_moe_deepseek_v3_grouped_router",
                "precision_moe_deepseek_router_correction_bias_load",
                "cuda_deepseek_router_route_api",
                "cuda_deepseek_v3_grouped_sigmoid_router_smoke",
                "cuda_hf_sequence_deepseek_v3_grouped_router_runtime",
                "cuda_hf_sequence_deepseek_native_layout_pack",
            ],
            &["run same-checkpoint full-layer routed output differential against /root/vllm"],
        ),
        (
            HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32,
            "deepseek_v3_split_fp8_expert_moe",
        ) => (
            "partial",
            &[
                "deepseek_routed_moe_reference",
                "deepseek_full_routed_moe_reference",
                "cuda_deepseek_routed_moe_api",
                "cuda_deepseek_routed_moe_smoke",
                "fp8_e4m3fn_e8m0_block_dequant_reference",
                "cuda_fp8_e4m3fn_e8m0_dequant_api",
                "cuda_fp8_e4m3fn_e8m0_block_dequant_smoke",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_v3_split_fp8_moe_runtime",
            ],
            &[
                "run same-checkpoint routed plus shared MoE output differential against /root/vllm",
                "benchmark fused MoE against vLLM FusedMoE",
            ],
        ),
        (HfArchitectureKind::DeepSeekV3, "deepseek_v3_mtp_optional") => (
            "optional_missing",
            &[],
            &["optional MTP draft layers are not implemented"],
        ),
        (HfArchitectureKind::DeepSeekV3, "deepseek_v3_vllm_e2e_parity") => (
            "partial",
            &[
                "deepseek_vllm_kv_cache_spec_planner",
                "deepseek_v3_manifest_uses_mla_fp8_scale_and_split_expert_names",
                "cuda_hf_sequence_deepseek_v3_mla_fullsize_kv_page_contents",
                "cuda_hf_sequence_deepseek_v3_grouped_router_runtime",
            ],
            &[
                "run same-checkpoint V3 greedy text differential against /root/vllm",
                "benchmark V3 prefill and decode throughput against /root/vllm on the same model and prompt",
            ],
        ),
        (HfArchitectureKind::DeepSeekV32, "deepseek_v32_vllm_e2e_parity") => (
            "partial",
            &[
                "deepseek_vllm_kv_cache_spec_planner",
                "deepseek_v32_manifest_adds_indexer_and_f32_norms",
                "cuda_hf_sequence_deepseek_packed_kv_footprint_accounting",
                "cuda_hf_sequence_deepseek_v32_fp8_ds_mla_page_contents",
                "cuda_hf_sequence_deepseek_v32_sparse_attention_output_hash",
            ],
            &[
                "run same-checkpoint V3.2 sparse MLA greedy text differential against /root/vllm",
                "benchmark V3.2 sparse MLA decode against /root/vllm on the same model and prompt",
            ],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_mhc_pre_post_head") => (
            "complete",
            &[
                "deepseek_v4_mhc_compressor_indexer_manifest",
                "deepseek_v4_mhc_warmup_plan_matches_vllm",
                "deepseek_v4_mhc_pre_post_head_reference_matches_vllm_torch",
                "cuda_deepseek_mhc_pre_api",
                "cuda_deepseek_mhc_pre_smoke",
                "cuda_deepseek_mhc_post_api",
                "cuda_deepseek_mhc_post_smoke",
                "cuda_deepseek_mhc_fused_post_pre_api",
                "cuda_deepseek_mhc_fused_post_pre_smoke",
                "cuda_deepseek_mhc_head_api",
                "cuda_deepseek_mhc_head_smoke",
                "deepseek_qkv_rmsnorm_reference",
                "cuda_deepseek_qkv_rmsnorm_api",
                "cuda_deepseek_qkv_rmsnorm_smoke",
                "cuda_deepseek_fused_inv_rope_fp8_quant_api",
                "cuda_deepseek_fused_inv_rope_fp8_quant_smoke",
                "cuda_hf_sequence_deepseek_descriptor_abi",
                "cuda_hf_sequence_deepseek_footprint_accounting",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_execution_guard",
                "cuda_hf_sequence_deepseek_v4_mhc_sequence_runtime",
                "cuda_hf_sequence_deepseek_v4_mhc_head_final_norm_runtime",
                "cuda_hf_sequence_deepseek_v4_mhc_native_profile_runtime",
            ],
            &[],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_mla_swa_cache") => (
            "partial",
            &[
                "deepseek_vllm_kv_cache_spec_planner",
                "deepseek_mla_decode_mqa_reference",
                "deepseek_mla_prefill_causal_mqa_reference",
                "cuda_deepseek_mla_decode_api",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_packed_kv_footprint_accounting",
                "cuda_hf_sequence_deepseek_v4_swa_window_runtime",
                "cuda_hf_sequence_deepseek_v4_swa_parallel_head_attention_runtime",
                "cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_page_runtime",
                "cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_page_contents",
                "cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_nonzero_page_contents",
                "cuda_hf_sequence_deepseek_v4_swa_fp8_ds_mla_fullsize_page_contents",
            ],
            &[
                "run full-size V4 SWA fp8_ds_mla page differential against vLLM FlashMLA",
                "replace per-head SWA attention kernel with the vLLM FlashMLA/FlashInfer tile scheduler",
            ],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_fp8_ds_mla_cache") => (
            "partial",
            &[
                "deepseek_vllm_kv_cache_spec_planner",
                "deepseek_mla_decode_mqa_reference",
                "deepseek_mla_prefill_causal_mqa_reference",
                "cuda_deepseek_mla_decode_api",
                "fp8_e4m3fn_e8m0_block_dequant_reference",
                "cuda_fp8_e4m3fn_e8m0_dequant_api",
                "cuda_fp8_e4m3fn_e8m0_block_dequant_smoke",
                "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_api",
                "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_token4_weight_reuse",
                "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_row8_token4_input_reuse",
                "cuda_deepseek_fp8_ds_mla_kv_pack_api",
                "cuda_deepseek_fp8_ds_mla_kv_pack_smoke",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_packed_kv_footprint_accounting",
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
                "deepseek_qkv_rmsnorm_reference",
                "cuda_deepseek_qkv_rmsnorm_api",
                "cuda_deepseek_qkv_rmsnorm_smoke",
                "deepseek_save_partial_states_reference",
                "cuda_deepseek_save_partial_states_api",
                "cuda_deepseek_save_partial_states_smoke",
                "deepseek_compress_norm_rope_fp8_cache_reference",
                "cuda_deepseek_compress_norm_rope_fp8_cache_api",
                "cuda_deepseek_compress_norm_rope_fp8_cache_smoke",
                "cuda_deepseek_compress_norm_rope_mxfp4_cache_api",
                "cuda_deepseek_compress_norm_rope_mxfp4_cache_smoke",
                "deepseek_compressed_slot_mapping_reference",
                "cuda_deepseek_compressed_slot_mapping_api",
                "cuda_deepseek_compressed_slot_mapping_smoke",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_packed_kv_footprint_accounting",
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
                "deepseek_compressed_slot_mapping_reference",
                "cuda_deepseek_compressed_slot_mapping_api",
                "cuda_deepseek_compressed_slot_mapping_smoke",
                "deepseek_c128_topk_metadata_reference",
                "cuda_deepseek_c128_topk_metadata_api",
                "cuda_deepseek_c128_topk_metadata_smoke",
                "deepseek_c4_indexer_topk_reference",
                "cuda_deepseek_c4_indexer_topk_api",
                "cuda_deepseek_c4_indexer_topk_smoke",
                "cuda_hf_sequence_deepseek_v4_index_topk_descriptor",
                "cuda_hf_sequence_deepseek_v4_compressed_scan_metrics",
                "cuda_hf_sequence_deepseek_v4_c4_sparse_topk_runtime",
                "cuda_hf_sequence_deepseek_v4_c4_sparse_topk_selection_hash",
                "cuda_hf_sequence_deepseek_v4_sparse_attention_swa_plus_topk",
                "cuda_hf_sequence_deepseek_v4_c4_topk_cover_all_shortcut",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_packed_kv_footprint_accounting",
            ],
            &[
                "run full-size V4 C4 sparse top-k differential against vLLM runtime",
                "integrate split paged-logits/top-k sparse scorer into full decode runtime for partial-cover cases",
            ],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_parallel_attention_gemm_streams") => (
            "partial",
            &[
                "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_api",
                "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_token4_weight_reuse",
                "cuda_fp8_e4m3fn_e8m0_scale_encoded_gemm_tokens_row8_token4_input_reuse",
                "cuda_hf_sequence_deepseek_v4_attention_aux_stream_resources",
                "cuda_hf_sequence_deepseek_v4_swa_parallel_head_attention_runtime",
                "cuda_hf_sequence_deepseek_v4_external_output_projection_runtime",
            ],
            &[
                "schedule attention GEMM/compressor/indexer kernels onto the V4 aux streams like vLLM",
                "replace external matvec output projection with vLLM DeepGEMM grouped o_proj",
                "measure stream overlap against vLLM DeepseekV4 attention",
            ],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_hash_and_bias_router") => (
            "partial",
            &[
                "deepseek_v4_hash_router_manifest",
                "deepseek_v4_sqrtsoftplus_hash_router_reference",
                "deepseek_v4_full_routed_moe_hash_reference",
                "precision_moe_deepseek_v4_sqrtsoftplus_router",
                "deepseek_v4_hash_route_table_i64_loader",
                "precision_moe_deepseek_v4_hash_route_table",
                "cuda_deepseek_router_route_api",
                "cuda_deepseek_v4_sqrtsoftplus_hash_router_smoke",
                "cuda_hf_sequence_deepseek_v4_bias_router_runtime",
                "cuda_hf_sequence_deepseek_v4_hash_router_runtime",
                "cuda_hf_sequence_deepseek_v4_sparse_moe_route_runtime",
                "cuda_hf_sequence_deepseek_native_layout_pack",
            ],
            &["run same-checkpoint full-layer routed output differential against /root/vllm"],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_megamoe_int8_fp4_experts") => (
            "partial",
            &[
                "mxfp4_e2m1_e8m0_block_dequant_reference",
                "cuda_mxfp4_e2m1_e8m0_dequant_api",
                "cuda_mxfp4_e2m1_e8m0_block_dequant_smoke",
                "cuda_deepseek_megamoe_prepare_api",
                "cuda_deepseek_megamoe_prepare_smoke",
                "cuda_deepseek_megamoe_eplb_mapping_api",
                "cuda_deepseek_megamoe_eplb_mapping_smoke",
                "cuda_deepseek_megamoe_fp8_fp4_expert_api",
                "cuda_deepseek_megamoe_fp8_fp4_expert_smoke",
                "deepseek_routed_moe_reference",
                "deepseek_full_routed_moe_reference",
                "cuda_deepseek_routed_moe_api",
                "cuda_deepseek_routed_moe_smoke",
                "cuda_hf_sequence_deepseek_native_layout_pack",
                "cuda_hf_sequence_deepseek_v4_mxfp4_expert_gate_up_runtime",
                "cuda_hf_sequence_deepseek_v4_mxfp4_expert_down_runtime",
                "cuda_hf_sequence_deepseek_v4_parallel_sparse_moe_runtime",
            ],
            &[
                "replace per-rank row-parallel MXFP4 decode expert kernels with DeepGEMM-equivalent batched MegaMoE kernels",
                "integrate dynamic EPLB rebalance and physical expert weight exchange into full runtime",
                "benchmark MegaMoE against vLLM deep_gemm_mega_moe/FusedMoE",
            ],
        ),
        (HfArchitectureKind::DeepSeekV4, "deepseek_v4_vllm_e2e_parity") => (
            "partial",
            &[
                "deepseek_v4_mhc_compressor_indexer_manifest",
                "deepseek_vllm_kv_cache_spec_planner",
                "cuda_hf_sequence_deepseek_packed_kv_footprint_accounting",
                "cuda_hf_sequence_deepseek_v4_mhc_sequence_runtime",
                "cuda_hf_sequence_deepseek_v4_fp8_ds_mla_fullsize_page_contents",
                "cuda_hf_sequence_deepseek_v4_c4_sparse_topk_runtime",
                "cuda_hf_sequence_deepseek_v4_hash_router_runtime",
            ],
            &[
                "run same-checkpoint V4 greedy text differential against /root/vllm",
                "benchmark V4 mHC, sparse MLA, and MegaMoE throughput against /root/vllm on the same model and prompt",
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
