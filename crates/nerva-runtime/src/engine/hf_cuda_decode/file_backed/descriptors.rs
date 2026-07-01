use std::ffi::CString;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::ffi::OsStrExt;

use nerva_core::types::dtype::DType;
use nerva_core::types::error::{NervaError, Result};
use nerva_cuda::decode::hf_chain::layer::{
    CUDA_HF_ATTENTION_DEEPSEEK_MLA, CUDA_HF_ATTENTION_FULL, CUDA_HF_ATTENTION_LINEAR_GDN,
    CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR, CUDA_HF_DEEPSEEK_FLAG_HASH_ROUTER, CUDA_HF_DEEPSEEK_FLAG_MOE,
    CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS, CUDA_HF_DEEPSEEK_FLAG_SLIDING_WINDOW,
    CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER, CUDA_HF_DEEPSEEK_MODE_V3_MLA,
    CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED, CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER,
    CUDA_HF_DEEPSEEK_MODE_V4_SWA, CUDA_HF_DEEPSEEK_MODE_V32_MLA_INDEXER,
    CUDA_HF_DEEPSEEK_ROPE_SCALING_DEEPSEEK, CUDA_HF_DEEPSEEK_ROPE_SCALING_NONE,
    CUDA_HF_DEEPSEEK_STORAGE_BF16, CUDA_HF_DEEPSEEK_STORAGE_FP8_SCALED, CUDA_HF_MLP_DENSE,
    CUDA_HF_MLP_SPARSE_MOE, CudaHfDecodeChainLayer, CudaHfDeepSeekLayer, CudaHfLinearGdnLayer,
};
use nerva_cuda::decode::hf_sequence::weight_plan::{
    CudaHfDecodeSequenceWeightBlock, hash_weight_blocks,
};
use nerva_model::hf::architecture::HfArchitectureKind;
use nerva_model::hf::deepseek_runtime::{
    DeepSeekAttentionExecutionKind, DeepSeekLayerExecution, deepseek_layer_execution_plan,
};
use nerva_model::hf::metadata::{HfAttentionLayerKind, HfMlpLayerKind, HfModelMetadata};
use nerva_model::weights::layout::entry::WeightBlockRole;
use nerva_model::weights::manifest::HfTensorManifest;
use nerva_model::weights::safetensors::shard::SafetensorsShardPlanEntry;

use crate::engine::hf_cuda_decode::contract::cuda_weight_descriptor_totals;
use crate::engine::hf_cuda_decode::descriptors::cuda_weight_strategy;
use crate::engine::hf_cuda_decode::file_backed::load::ShardBackedWeights;
use crate::engine::hf_cuda_decode::resident::{
    cuda_compute_capability, default_large_file_backed_hotset_bytes, strategy_bytes,
};
use crate::engine::hf_cuda_decode::summary::HfCudaResidentWeightSummary;
use crate::engine::runtime::Runtime;
use crate::residency::budget::ResidencyBudget;
use crate::weights::execution::strategy::ResidentWeightExecutionStrategy;

const EMPTY: [u16; 0] = [];
const MARKER: [u16; 1] = [0];

pub(super) struct ShardBackedResidentWeights {
    pub summary: HfCudaResidentWeightSummary,
    pub descriptors: Vec<CudaHfDecodeSequenceWeightBlock>,
    pub _source_paths: Vec<CString>,
    pub _host_buffers: Vec<Vec<u16>>,
}

#[cfg(test)]
pub(super) fn descriptor_marker_layers(
    metadata: &HfModelMetadata,
) -> Result<Vec<CudaHfDecodeChainLayer<'static>>> {
    descriptor_marker_layers_with_storage(metadata, CUDA_HF_DEEPSEEK_STORAGE_FP8_SCALED)
}

pub(super) fn descriptor_marker_layers_for_manifest(
    metadata: &HfModelMetadata,
    manifest: &HfTensorManifest,
) -> Result<Vec<CudaHfDecodeChainLayer<'static>>> {
    descriptor_marker_layers_with_storage(
        metadata,
        deepseek_storage_from_manifest(metadata, manifest),
    )
}

fn descriptor_marker_layers_with_storage(
    metadata: &HfModelMetadata,
    deepseek_storage: u32,
) -> Result<Vec<CudaHfDecodeChainLayer<'static>>> {
    let qk_norm = metadata.qk_norm.then_some(&MARKER[..]);
    let qkv_bias = metadata.attention_qkv_bias.then_some(&MARKER[..]);
    let output_bias = metadata.attention_output_bias.then_some(&MARKER[..]);
    let deepseek_plan = if metadata.architecture.is_deepseek() {
        Some(deepseek_layer_execution_plan(metadata)?)
    } else {
        None
    };
    (0..metadata.num_hidden_layers)
        .map(|layer| {
            let deepseek_execution = deepseek_plan
                .as_ref()
                .and_then(|plan| plan.layers.get(layer));
            let deepseek = deepseek_execution
                .map(|execution| deepseek_marker(metadata, execution, deepseek_storage))
                .transpose()?;
            let sparse_moe = metadata
                .mlp_layer_types
                .get(layer)
                .is_some_and(|kind| *kind == HfMlpLayerKind::SparseMoe);
            let q_gate = matches!(
                metadata.architecture,
                HfArchitectureKind::Qwen35 | HfArchitectureKind::Qwen35Moe
            ) && metadata
                .attention_layer_types
                .get(layer)
                .is_some_and(|kind| *kind == HfAttentionLayerKind::Full);
            let attention_kind = if deepseek.is_some() {
                CUDA_HF_ATTENTION_DEEPSEEK_MLA
            } else if metadata
                .attention_layer_types
                .get(layer)
                .is_some_and(|kind| *kind == HfAttentionLayerKind::Linear)
            {
                CUDA_HF_ATTENTION_LINEAR_GDN
            } else {
                CUDA_HF_ATTENTION_FULL
            };
            Ok(CudaHfDecodeChainLayer {
                rms_attn_weight: &EMPTY,
                rms_mlp_weight: &EMPTY,
                w_q: &EMPTY,
                w_q_gate: q_gate.then_some(&MARKER[..]),
                w_k: &EMPTY,
                q_norm_weight: qk_norm,
                k_norm_weight: qk_norm,
                w_v: &EMPTY,
                w_o: &EMPTY,
                q_bias: qkv_bias,
                k_bias: qkv_bias,
                v_bias: qkv_bias,
                o_bias: output_bias,
                w_gate: &EMPTY,
                w_up: &EMPTY,
                w_down: &EMPTY,
                w_router: None,
                w_expert_gate_up: None,
                w_expert_down: None,
                w_shared_expert_gate: None,
                w_shared_expert_up: None,
                w_shared_expert_down: None,
                w_shared_expert_router: None,
                linear_gdn: deepseek
                    .is_none()
                    .then(|| linear_gdn_marker(metadata, layer))
                    .flatten(),
                deepseek,
                mlp_kind: if sparse_moe {
                    CUDA_HF_MLP_SPARSE_MOE
                } else {
                    CUDA_HF_MLP_DENSE
                },
                moe_intermediate: if sparse_moe {
                    metadata.moe_intermediate_size.unwrap_or(0)
                } else {
                    0
                },
                shared_expert_intermediate: if sparse_moe {
                    metadata.shared_expert_intermediate_size.unwrap_or(0)
                } else {
                    0
                },
                num_experts: if sparse_moe {
                    metadata.num_experts.unwrap_or(0)
                } else {
                    0
                },
                experts_per_token: if sparse_moe {
                    metadata.num_experts_per_tok.unwrap_or(0)
                } else {
                    0
                },
                norm_topk_prob: sparse_moe && metadata.norm_topk_prob,
                attention_kind,
            })
        })
        .collect()
}

fn deepseek_storage_from_manifest(metadata: &HfModelMetadata, manifest: &HfTensorManifest) -> u32 {
    if !matches!(
        metadata.architecture,
        HfArchitectureKind::DeepSeekV3 | HfArchitectureKind::DeepSeekV32
    ) {
        return CUDA_HF_DEEPSEEK_STORAGE_FP8_SCALED;
    }
    if manifest
        .entries
        .iter()
        .any(|entry| entry.role == WeightBlockRole::DeepSeekQALoraScaleInv)
    {
        return CUDA_HF_DEEPSEEK_STORAGE_FP8_SCALED;
    }
    manifest
        .entries
        .iter()
        .find(|entry| {
            entry.role == WeightBlockRole::DeepSeekQALoraProjection && entry.layer == Some(0)
        })
        .filter(|entry| entry.dtype == DType::BF16)
        .map(|_| CUDA_HF_DEEPSEEK_STORAGE_BF16)
        .unwrap_or(CUDA_HF_DEEPSEEK_STORAGE_FP8_SCALED)
}

fn deepseek_marker(
    metadata: &HfModelMetadata,
    execution: &DeepSeekLayerExecution,
    storage: u32,
) -> Result<CudaHfDeepSeekLayer> {
    let mode = match execution.attention_kind {
        DeepSeekAttentionExecutionKind::V3Mla => CUDA_HF_DEEPSEEK_MODE_V3_MLA,
        DeepSeekAttentionExecutionKind::V32MlaWithIndexer => CUDA_HF_DEEPSEEK_MODE_V32_MLA_INDEXER,
        DeepSeekAttentionExecutionKind::V4SlidingWindowMla => CUDA_HF_DEEPSEEK_MODE_V4_SWA,
        DeepSeekAttentionExecutionKind::V4CompressedMla => CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED,
        DeepSeekAttentionExecutionKind::V4CompressedMlaWithSparseIndexer => {
            CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER
        }
    };
    let mut flags = 0u32;
    if execution.uses_sparse_indexer {
        flags |= CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER;
    }
    if execution.uses_compressor {
        flags |= CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR;
    }
    if execution.uses_hash_router {
        flags |= CUDA_HF_DEEPSEEK_FLAG_HASH_ROUTER;
    }
    if execution.uses_moe {
        flags |= CUDA_HF_DEEPSEEK_FLAG_MOE;
    }
    if execution.uses_sliding_window_cache {
        flags |= CUDA_HF_DEEPSEEK_FLAG_SLIDING_WINDOW;
    }
    if execution.uses_moe
        && !execution.uses_hash_router
        && metadata.topk_method.as_deref() == Some("noaux_tc")
    {
        flags |= CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS;
    }
    let (
        rope_scaling_type,
        rope_original_max_position,
        rope_scaling_factor,
        rope_extrapolation_factor,
        rope_attn_factor,
        rope_beta_fast,
        rope_beta_slow,
        rope_mscale,
        rope_mscale_all_dim,
    ) = deepseek_rope_scaling_fields(metadata)?;

    Ok(CudaHfDeepSeekLayer {
        mode,
        flags,
        storage,
        hc_mult: metadata.hc_mult.unwrap_or(0),
        hc_sinkhorn_iters: metadata.hc_sinkhorn_iters.unwrap_or(0),
        q_lora_rank: required_deepseek_usize(metadata.q_lora_rank, "q_lora_rank")?,
        kv_lora_rank: metadata.kv_lora_rank.unwrap_or(0),
        o_lora_rank: metadata.o_lora_rank.unwrap_or(0),
        o_groups: metadata.o_groups.unwrap_or(0),
        qk_nope_head_dim: required_deepseek_usize(metadata.qk_nope_head_dim, "qk_nope_head_dim")?,
        qk_rope_head_dim: required_deepseek_usize(metadata.qk_rope_head_dim, "qk_rope_head_dim")?,
        v_head_dim: required_deepseek_usize(metadata.v_head_dim, "v_head_dim")?,
        compress_ratio: execution.compress_ratio,
        index_topk: execution.index_topk,
        index_n_heads: metadata.index_n_heads.unwrap_or(0),
        index_head_dim: metadata.index_head_dim.unwrap_or(0),
        router_num_groups: metadata.num_expert_groups.unwrap_or(0),
        router_topk_groups: metadata.topk_group.unwrap_or(0),
        routed_scaling_factor: metadata.routed_scaling_factor.unwrap_or(1.0),
        hc_eps: metadata.hc_eps.unwrap_or(0.0),
        hc_post_alpha: if matches!(
            execution.attention_kind,
            DeepSeekAttentionExecutionKind::V4SlidingWindowMla
                | DeepSeekAttentionExecutionKind::V4CompressedMla
                | DeepSeekAttentionExecutionKind::V4CompressedMlaWithSparseIndexer
        ) {
            2.0
        } else {
            0.0
        },
        rope_scaling_type,
        rope_original_max_position,
        rope_scaling_factor,
        rope_extrapolation_factor,
        rope_attn_factor,
        rope_beta_fast,
        rope_beta_slow,
        rope_mscale,
        rope_mscale_all_dim,
        compress_rope_theta: metadata.compress_rope_theta,
        swiglu_limit: metadata.swiglu_limit,
    })
}

fn deepseek_rope_scaling_fields(
    metadata: &HfModelMetadata,
) -> Result<(u32, usize, f32, f32, f32, f32, f32, f32, f32)> {
    let Some(rope_scaling) = metadata.rope_scaling.as_ref() else {
        return Ok((
            CUDA_HF_DEEPSEEK_ROPE_SCALING_NONE,
            0,
            0.0,
            1.0,
            1.0,
            32.0,
            1.0,
            1.0,
            0.0,
        ));
    };
    if rope_scaling.rope_type != "deepseek_yarn"
        && rope_scaling.rope_type != "deepseek_llama_scaling"
    {
        return Err(NervaError::InvalidArgument {
            reason: format!(
                "DeepSeek CUDA descriptor unsupported rope scaling type {}",
                rope_scaling.rope_type
            ),
        });
    }
    let factor = rope_scaling
        .factor
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "DeepSeek scaled RoPE is missing factor".to_string(),
        })?;
    let original_max = rope_scaling
        .original_max_position_embeddings
        .ok_or_else(|| NervaError::InvalidArgument {
            reason: "DeepSeek scaled RoPE is missing original_max_position_embeddings".to_string(),
        })?;
    let default_mscale = if metadata.architecture == HfArchitectureKind::DeepSeekV4 {
        0.0
    } else {
        1.0
    };
    Ok((
        CUDA_HF_DEEPSEEK_ROPE_SCALING_DEEPSEEK,
        original_max,
        factor,
        rope_scaling.extrapolation_factor.unwrap_or(1.0),
        rope_scaling.attn_factor.unwrap_or(1.0),
        rope_scaling.beta_fast.unwrap_or(32.0),
        rope_scaling.beta_slow.unwrap_or(1.0),
        rope_scaling.mscale.unwrap_or(default_mscale),
        rope_scaling.mscale_all_dim.unwrap_or(0.0),
    ))
}

fn required_deepseek_usize(value: Option<usize>, name: &'static str) -> Result<usize> {
    value.ok_or_else(|| NervaError::InvalidArgument {
        reason: format!("DeepSeek CUDA descriptor metadata is missing {name}"),
    })
}

fn linear_gdn_marker(
    metadata: &HfModelMetadata,
    layer: usize,
) -> Option<CudaHfLinearGdnLayer<'static>> {
    if !metadata
        .attention_layer_types
        .get(layer)
        .is_some_and(|kind| *kind == HfAttentionLayerKind::Linear)
    {
        return None;
    }
    Some(CudaHfLinearGdnLayer {
        key_heads: metadata.linear_num_key_heads.unwrap_or(0),
        value_heads: metadata.linear_num_value_heads.unwrap_or(0),
        key_head_dim: metadata.linear_key_head_dim.unwrap_or(0),
        value_head_dim: metadata.linear_value_head_dim.unwrap_or(0),
        conv_kernel: metadata.linear_conv_kernel_dim.unwrap_or(0),
        w_conv: &MARKER,
        w_qkv: &MARKER,
        w_z: &MARKER,
        w_b: &MARKER,
        w_a: &MARKER,
        dt_bias: &MARKER,
        a_log: &[],
        norm_weight: &MARKER,
        w_out: &MARKER,
    })
}

pub(super) fn shard_backed_resident_weights(
    runtime: &Runtime,
    weights: &ShardBackedWeights,
    compute_capability: Option<u32>,
) -> Result<ShardBackedResidentWeights> {
    let compute_capability = compute_capability.or_else(cuda_compute_capability);
    let manifest = &weights.manifest;
    let hotset_bytes = default_large_file_backed_hotset_bytes(manifest.total_weight_bytes);
    let budget = ResidencyBudget::new(hotset_bytes, 0, manifest.total_weight_bytes);
    let mut table = runtime.materialize_hf_weight_manifest_with_budget(manifest, budget)?;
    let hotset = runtime.promote_resident_weight_hotset(&mut table, hotset_bytes)?;
    let plan = runtime.plan_resident_weight_execution(
        &table,
        weights.manifest.entries.len(),
        compute_capability,
    )?;
    let run = runtime.execute_resident_weight_execution_plan(&table, &plan)?;
    let DescriptorTable {
        descriptors,
        source_paths,
        host_buffers,
    } = cuda_weight_descriptors(weights, &plan)?;
    let descriptor_hash = hash_weight_blocks(&descriptors);
    let fallback_bytes = strategy_bytes(&plan, ResidentWeightExecutionStrategy::CpuExactFallback);
    let descriptor_totals = cuda_weight_descriptor_totals(&descriptors);
    Ok(ShardBackedResidentWeights {
        summary: HfCudaResidentWeightSummary {
            plan_steps: plan.steps.len() as u64,
            plan_weight_bytes: descriptor_totals.weight_bytes,
            plan_descriptor_blocks: descriptors.len() as u64,
            plan_descriptor_hash: descriptor_hash,
            hotset_promoted_blocks: hotset.promoted_blocks as u64,
            hotset_promoted_bytes: hotset.promoted_bytes as u64,
            hotset_kept_dram_blocks: hotset.kept_dram_blocks as u64,
            plan_gpu_resident_weight_bytes: descriptor_totals.gpu_resident_weight_bytes,
            plan_gpu_staged_weight_bytes: descriptor_totals.gpu_staged_weight_bytes,
            plan_fallback_weight_bytes: fallback_bytes,
            plan_gpu_resident_steps: plan.gpu_resident_steps,
            plan_gpu_staged_steps: plan.gpu_staged_steps,
            plan_fallback_steps: plan.fallback_steps,
            plan_block_version_dependencies: plan.block_version_dependencies,
            run_steps: run.steps as u64,
            run_gpu_resident_steps: run.gpu_resident_steps,
            run_gpu_staged_steps: run.gpu_staged_steps,
            run_fallback_steps: run.fallback_steps,
            run_block_version_dependencies: run.block_version_dependencies,
            hot_path_allocations: hotset.hot_path_allocations
                + run.hot_path_allocations
                + plan.ledger.hot_path_allocations,
            ..HfCudaResidentWeightSummary::default()
        },
        descriptors,
        _source_paths: source_paths,
        _host_buffers: host_buffers,
    })
}

struct DescriptorTable {
    descriptors: Vec<CudaHfDecodeSequenceWeightBlock>,
    source_paths: Vec<CString>,
    host_buffers: Vec<Vec<u16>>,
}

fn cuda_weight_descriptors(
    weights: &ShardBackedWeights,
    plan: &crate::weights::execution::plan::ResidentWeightExecutionPlan,
) -> Result<DescriptorTable> {
    if plan.steps.len() != weights.manifest.entries.len()
        || plan.steps.len() != weights.shard_plan.entries.len()
    {
        return Err(NervaError::InvalidArgument {
            reason: "CUDA shard-backed descriptor counts do not match".to_string(),
        });
    }
    let mut offset_bytes = 0u64;
    let mut descriptors = Vec::with_capacity(plan.steps.len());
    let mut source_paths = Vec::with_capacity(plan.steps.len());
    let mut host_buffers: Vec<Vec<u16>> = Vec::new();
    for ((step, manifest), shard) in plan
        .steps
        .iter()
        .zip(&weights.manifest.entries)
        .zip(&weights.shard_plan.entries)
    {
        if step.name != manifest.name || shard.tensor_name != manifest.name {
            return Err(NervaError::InvalidArgument {
                reason: "CUDA shard-backed descriptor order does not match manifest".to_string(),
            });
        }
        let source_path = weights.source_path(shard)?;
        let descriptor_bytes = shard.bytes.next_multiple_of(2);
        let (host_source, source_file, source_file_len, file_offset_begin) = if shard.bytes % 2 == 0
        {
            let source_path = CString::new(source_path.as_os_str().as_bytes()).map_err(|_| {
                NervaError::InvalidArgument {
                    reason: format!(
                        "safetensors shard path for {} contains a nul byte",
                        shard.tensor_name
                    ),
                }
            })?;
            source_paths.push(source_path);
            let source_path = source_paths.last().expect("source path was just pushed");
            (
                std::ptr::null(),
                source_path.as_ptr(),
                source_path.as_bytes().len() as u64,
                shard.file_offset_begin as u64,
            )
        } else {
            let padded = read_padded_u16_tensor(&source_path, shard)?;
            host_buffers.push(padded);
            let host = host_buffers
                .last()
                .expect("host buffer was just pushed")
                .as_ptr();
            (host, std::ptr::null(), 0, 0)
        };
        descriptors.push(CudaHfDecodeSequenceWeightBlock {
            host_source,
            source_file,
            source_file_len,
            file_offset_begin,
            block_id: step.block_id.0,
            block_version: step.block_version,
            offset_bytes,
            bytes: descriptor_bytes as u64,
            strategy: cuda_weight_strategy(step.strategy)?,
            reserved: 0,
        });
        offset_bytes = offset_bytes
            .checked_add(descriptor_bytes as u64)
            .ok_or_else(|| NervaError::AllocationFailed {
                bytes: descriptor_bytes,
                reason: "CUDA shard-backed descriptor offset overflow".to_string(),
            })?;
    }
    Ok(DescriptorTable {
        descriptors,
        source_paths,
        host_buffers,
    })
}

fn read_padded_u16_tensor(
    source_path: &std::path::Path,
    shard: &SafetensorsShardPlanEntry,
) -> Result<Vec<u16>> {
    let mut file = File::open(source_path).map_err(|err| NervaError::InvalidArgument {
        reason: format!(
            "failed to open safetensors shard {}: {err}",
            source_path.display()
        ),
    })?;
    file.seek(SeekFrom::Start(shard.file_offset_begin as u64))
        .map_err(|err| NervaError::InvalidArgument {
            reason: format!(
                "failed to seek safetensors shard {}: {err}",
                source_path.display()
            ),
        })?;
    let mut bytes = vec![0u8; shard.bytes.next_multiple_of(2)];
    file.read_exact(&mut bytes[..shard.bytes])
        .map_err(|err| NervaError::InvalidArgument {
            reason: format!(
                "failed to read safetensors tensor {} from {}: {err}",
                shard.tensor_name,
                source_path.display()
            ),
        })?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use nerva_core::types::dtype::DType;
    use nerva_cuda::decode::hf_chain::layer::{
        CUDA_HF_ATTENTION_DEEPSEEK_MLA, CUDA_HF_ATTENTION_FULL, CUDA_HF_ATTENTION_LINEAR_GDN,
        CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR, CUDA_HF_DEEPSEEK_FLAG_HASH_ROUTER,
        CUDA_HF_DEEPSEEK_FLAG_MOE, CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS,
        CUDA_HF_DEEPSEEK_FLAG_SLIDING_WINDOW, CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER,
        CUDA_HF_DEEPSEEK_MODE_V3_MLA, CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED,
        CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER, CUDA_HF_DEEPSEEK_MODE_V4_SWA,
        CUDA_HF_DEEPSEEK_MODE_V32_MLA_INDEXER, CUDA_HF_DEEPSEEK_ROPE_SCALING_DEEPSEEK,
        CUDA_HF_DEEPSEEK_STORAGE_BF16, CUDA_HF_MLP_DENSE, CUDA_HF_MLP_SPARSE_MOE,
    };
    use nerva_model::hf::architecture::HfArchitectureKind;
    use nerva_model::hf::metadata::{
        HfAttentionLayerKind, HfMlpLayerKind, HfModelMetadata, HfRopeScalingMetadata,
    };
    use nerva_model::weights::layout::plan::plan_hf_weight_layout_for_safetensors_index;
    use nerva_model::weights::manifest::build_hf_tensor_manifest;

    use super::{descriptor_marker_layers, descriptor_marker_layers_for_manifest};

    #[test]
    fn descriptor_marker_layers_preserve_sparse_moe_metadata() {
        let metadata = HfModelMetadata {
            architecture: HfArchitectureKind::Qwen3Moe,
            hidden_size: 4,
            num_hidden_layers: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: 8,
            vocab_size: 16,
            max_position_embeddings: None,
            sliding_window: None,
            rope_theta: None,
            rope_scaling: None,
            compress_rope_theta: None,
            rms_norm_eps: Some(1e-5),
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: false,
            hidden_act: Some("silu".to_string()),
            attention_bias: false,
            attention_qkv_bias: false,
            attention_output_bias: false,
            qk_norm: true,
            mlp_bias: false,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            attention_layer_types: vec![HfAttentionLayerKind::Full; 2],
            mlp_layer_types: vec![HfMlpLayerKind::SparseMoe, HfMlpLayerKind::Dense],
            moe_intermediate_size: Some(3),
            shared_expert_intermediate_size: None,
            num_experts: Some(4),
            num_experts_per_tok: Some(2),
            decoder_sparse_step: Some(1),
            norm_topk_prob: true,
            moe_first_k_dense_replace: None,
            moe_layer_freq: None,
            num_expert_groups: None,
            topk_group: None,
            topk_method: None,
            scoring_func: None,
            routed_scaling_factor: None,
            q_lora_rank: None,
            kv_lora_rank: None,
            o_lora_rank: None,
            o_groups: None,
            qk_nope_head_dim: None,
            qk_rope_head_dim: None,
            v_head_dim: None,
            index_topk: None,
            index_topk_freq: None,
            index_skip_topk_offset: None,
            index_topk_pattern: Vec::new(),
            index_n_heads: None,
            index_head_dim: None,
            compress_ratios: Vec::new(),
            hc_mult: None,
            hc_sinkhorn_iters: None,
            hc_eps: None,
            num_nextn_predict_layers: None,
            num_hash_layers: None,
            swiglu_limit: None,
            expert_dtype: None,
            torch_dtype: Some(DType::F16),
        };

        let layers = descriptor_marker_layers(&metadata).unwrap();

        assert_eq!(layers[0].mlp_kind, CUDA_HF_MLP_SPARSE_MOE);
        assert_eq!(layers[0].moe_intermediate, 3);
        assert_eq!(layers[0].num_experts, 4);
        assert_eq!(layers[0].experts_per_token, 2);
        assert!(layers[0].norm_topk_prob);
        assert_eq!(layers[1].mlp_kind, CUDA_HF_MLP_DENSE);
        assert_eq!(layers[1].moe_intermediate, 0);
        assert_eq!(layers[1].num_experts, 0);
        assert!(layers[0].w_q_gate.is_none());
        assert!(layers[1].w_q_gate.is_none());
        assert_eq!(layers[0].attention_kind, CUDA_HF_ATTENTION_FULL);
        assert_eq!(layers[1].attention_kind, CUDA_HF_ATTENTION_FULL);
    }

    #[test]
    fn descriptor_marker_layers_reserve_qwen35_full_attention_q_gate() {
        let metadata = HfModelMetadata {
            architecture: HfArchitectureKind::Qwen35Moe,
            hidden_size: 4,
            num_hidden_layers: 3,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: 8,
            vocab_size: 16,
            max_position_embeddings: None,
            sliding_window: None,
            rope_theta: None,
            rope_scaling: None,
            compress_rope_theta: None,
            rms_norm_eps: Some(1e-5),
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: false,
            hidden_act: Some("silu".to_string()),
            attention_bias: false,
            attention_qkv_bias: false,
            attention_output_bias: false,
            qk_norm: true,
            mlp_bias: false,
            linear_conv_kernel_dim: Some(4),
            linear_key_head_dim: Some(128),
            linear_value_head_dim: Some(128),
            linear_num_key_heads: Some(16),
            linear_num_value_heads: Some(32),
            attention_layer_types: vec![
                HfAttentionLayerKind::Linear,
                HfAttentionLayerKind::Full,
                HfAttentionLayerKind::Full,
            ],
            mlp_layer_types: vec![HfMlpLayerKind::SparseMoe; 3],
            moe_intermediate_size: Some(3),
            shared_expert_intermediate_size: None,
            num_experts: Some(4),
            num_experts_per_tok: Some(2),
            decoder_sparse_step: Some(1),
            norm_topk_prob: true,
            moe_first_k_dense_replace: None,
            moe_layer_freq: None,
            num_expert_groups: None,
            topk_group: None,
            topk_method: None,
            scoring_func: None,
            routed_scaling_factor: None,
            q_lora_rank: None,
            kv_lora_rank: None,
            o_lora_rank: None,
            o_groups: None,
            qk_nope_head_dim: None,
            qk_rope_head_dim: None,
            v_head_dim: None,
            index_topk: None,
            index_topk_freq: None,
            index_skip_topk_offset: None,
            index_topk_pattern: Vec::new(),
            index_n_heads: None,
            index_head_dim: None,
            compress_ratios: Vec::new(),
            hc_mult: None,
            hc_sinkhorn_iters: None,
            hc_eps: None,
            num_nextn_predict_layers: None,
            num_hash_layers: None,
            swiglu_limit: None,
            expert_dtype: None,
            torch_dtype: Some(DType::F16),
        };

        let layers = descriptor_marker_layers(&metadata).unwrap();

        assert!(layers[0].w_q_gate.is_none());
        assert!(layers[1].w_q_gate.is_some());
        assert!(layers[2].w_q_gate.is_some());
        assert_eq!(layers[0].attention_kind, CUDA_HF_ATTENTION_LINEAR_GDN);
        assert_eq!(layers[1].attention_kind, CUDA_HF_ATTENTION_FULL);
        assert_eq!(layers[2].attention_kind, CUDA_HF_ATTENTION_FULL);
    }

    #[test]
    fn descriptor_marker_layers_preserve_deepseek_v3_mla_metadata() {
        let mut metadata = base_metadata(HfArchitectureKind::DeepSeekV3, 2);
        metadata.num_attention_heads = 128;
        metadata.num_key_value_heads = 128;
        metadata.head_dim = 192;
        metadata.intermediate_size = 4096;
        metadata.mlp_layer_types = vec![HfMlpLayerKind::Dense, HfMlpLayerKind::SparseMoe];
        metadata.moe_intermediate_size = Some(2048);
        metadata.shared_expert_intermediate_size = Some(2048);
        metadata.num_experts = Some(256);
        metadata.num_experts_per_tok = Some(8);
        metadata.norm_topk_prob = true;
        metadata.topk_method = Some("noaux_tc".to_string());
        metadata.q_lora_rank = Some(1536);
        metadata.kv_lora_rank = Some(512);
        metadata.qk_nope_head_dim = Some(128);
        metadata.qk_rope_head_dim = Some(64);
        metadata.v_head_dim = Some(128);
        metadata.torch_dtype = Some(DType::BF16);

        let layers = descriptor_marker_layers(&metadata).unwrap();

        assert_eq!(layers[0].attention_kind, CUDA_HF_ATTENTION_DEEPSEEK_MLA);
        assert_eq!(layers[1].attention_kind, CUDA_HF_ATTENTION_DEEPSEEK_MLA);
        let first = layers[0].deepseek.unwrap();
        assert_eq!(first.mode, CUDA_HF_DEEPSEEK_MODE_V3_MLA);
        assert_eq!(first.flags, 0);
        assert_eq!(first.q_lora_rank, 1536);
        assert_eq!(first.kv_lora_rank, 512);
        assert_eq!(first.qk_nope_head_dim, 128);
        assert_eq!(first.qk_rope_head_dim, 64);
        assert_eq!(first.v_head_dim, 128);
        assert_eq!(first.compress_ratio, 1);
        assert_eq!(
            layers[1].deepseek.unwrap().flags,
            CUDA_HF_DEEPSEEK_FLAG_MOE | CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS
        );
    }

    #[test]
    fn descriptor_marker_layers_detect_deepseek_v3_bf16_manifest_storage() {
        let mut metadata = base_metadata(HfArchitectureKind::DeepSeekV3, 2);
        metadata.num_attention_heads = 128;
        metadata.num_key_value_heads = 128;
        metadata.head_dim = 192;
        metadata.intermediate_size = 4096;
        metadata.mlp_layer_types = vec![HfMlpLayerKind::Dense, HfMlpLayerKind::SparseMoe];
        metadata.moe_intermediate_size = Some(2048);
        metadata.shared_expert_intermediate_size = Some(2048);
        metadata.num_experts = Some(256);
        metadata.num_experts_per_tok = Some(8);
        metadata.topk_method = Some("noaux_tc".to_string());
        metadata.q_lora_rank = Some(1536);
        metadata.kv_lora_rank = Some(512);
        metadata.qk_nope_head_dim = Some(128);
        metadata.qk_rope_head_dim = Some(64);
        metadata.v_head_dim = Some(128);
        metadata.qk_norm = false;
        metadata.torch_dtype = Some(DType::BF16);
        let index = r#"{"metadata":{"total_size":1},"weight_map":{"model.layers.0.self_attn.q_a_proj.weight":"model-00001-of-00001.safetensors"}}"#;
        let manifest = build_hf_tensor_manifest(
            &plan_hf_weight_layout_for_safetensors_index(&metadata, index).unwrap(),
        )
        .unwrap();

        let layers = descriptor_marker_layers_for_manifest(&metadata, &manifest).unwrap();

        assert_eq!(
            layers[0].deepseek.unwrap().storage,
            CUDA_HF_DEEPSEEK_STORAGE_BF16
        );
        assert_eq!(
            layers[1].deepseek.unwrap().storage,
            CUDA_HF_DEEPSEEK_STORAGE_BF16
        );
    }

    #[test]
    fn descriptor_marker_layers_preserve_deepseek_v4_mla_modes() {
        let mut metadata = base_metadata(HfArchitectureKind::DeepSeekV4, 4);
        metadata.hidden_size = 4096;
        metadata.num_attention_heads = 64;
        metadata.num_key_value_heads = 1;
        metadata.head_dim = 512;
        metadata.intermediate_size = 4096;
        metadata.sliding_window = Some(4096);
        metadata.mlp_layer_types = vec![HfMlpLayerKind::SparseMoe; 4];
        metadata.moe_intermediate_size = Some(2048);
        metadata.shared_expert_intermediate_size = Some(2048);
        metadata.num_experts = Some(256);
        metadata.num_experts_per_tok = Some(6);
        metadata.norm_topk_prob = true;
        metadata.q_lora_rank = Some(1024);
        metadata.kv_lora_rank = Some(512);
        metadata.o_lora_rank = Some(1024);
        metadata.o_groups = Some(8);
        metadata.qk_nope_head_dim = Some(448);
        metadata.qk_rope_head_dim = Some(64);
        metadata.v_head_dim = Some(512);
        metadata.index_topk = Some(512);
        metadata.index_n_heads = Some(64);
        metadata.index_head_dim = Some(128);
        metadata.compress_ratios = vec![0, 1, 4, 128];
        metadata.hc_mult = Some(4);
        metadata.num_hash_layers = Some(3);
        metadata.topk_method = Some("noaux_tc".to_string());
        metadata.scoring_func = Some("sqrtsoftplus".to_string());
        metadata.rope_scaling = Some(HfRopeScalingMetadata {
            rope_type: "deepseek_yarn".to_string(),
            factor: Some(40.0),
            original_max_position_embeddings: Some(4096),
            extrapolation_factor: None,
            attn_factor: None,
            beta_fast: None,
            beta_slow: None,
            mscale: None,
            mscale_all_dim: None,
        });
        metadata.compress_rope_theta = Some(1_000_000.0);
        metadata.swiglu_limit = Some(10.0);
        metadata.expert_dtype = Some("fp4".to_string());
        metadata.torch_dtype = Some(DType::BF16);

        let layers = descriptor_marker_layers(&metadata).unwrap();

        assert_eq!(
            layers
                .iter()
                .map(|layer| layer.deepseek.unwrap().mode)
                .collect::<Vec<_>>(),
            vec![
                CUDA_HF_DEEPSEEK_MODE_V4_SWA,
                CUDA_HF_DEEPSEEK_MODE_V4_SWA,
                CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED_INDEXER,
                CUDA_HF_DEEPSEEK_MODE_V4_COMPRESSED,
            ]
        );
        assert_eq!(
            layers
                .iter()
                .map(|layer| layer.deepseek.unwrap().compress_ratio)
                .collect::<Vec<_>>(),
            vec![1, 1, 4, 128]
        );
        assert_eq!(
            layers[0].deepseek.unwrap().flags,
            CUDA_HF_DEEPSEEK_FLAG_SLIDING_WINDOW
                | CUDA_HF_DEEPSEEK_FLAG_HASH_ROUTER
                | CUDA_HF_DEEPSEEK_FLAG_MOE
        );
        assert_eq!(
            layers[2].deepseek.unwrap().flags,
            CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR
                | CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER
                | CUDA_HF_DEEPSEEK_FLAG_HASH_ROUTER
                | CUDA_HF_DEEPSEEK_FLAG_MOE
        );
        assert_eq!(
            layers[3].deepseek.unwrap().flags,
            CUDA_HF_DEEPSEEK_FLAG_COMPRESSOR
                | CUDA_HF_DEEPSEEK_FLAG_MOE
                | CUDA_HF_DEEPSEEK_FLAG_ROUTER_BIAS
        );
        assert_eq!(layers[3].deepseek.unwrap().hc_mult, 4);
        assert_eq!(layers[2].deepseek.unwrap().index_n_heads, 64);
        assert_eq!(layers[2].deepseek.unwrap().index_head_dim, 128);
        assert_eq!(
            layers[2].deepseek.unwrap().compress_rope_theta,
            Some(1_000_000.0)
        );
        assert_eq!(
            layers[2].deepseek.unwrap().rope_scaling_type,
            CUDA_HF_DEEPSEEK_ROPE_SCALING_DEEPSEEK
        );
        assert_eq!(layers[2].deepseek.unwrap().rope_scaling_factor, 40.0);
        assert_eq!(layers[2].deepseek.unwrap().rope_mscale, 0.0);
        assert_eq!(
            layers[3].deepseek.unwrap().compress_rope_theta,
            Some(1_000_000.0)
        );
        assert_eq!(layers[0].deepseek.unwrap().swiglu_limit, Some(10.0));
        assert_eq!(layers[3].deepseek.unwrap().swiglu_limit, Some(10.0));
    }

    #[test]
    fn descriptor_marker_layers_apply_deepseek_v32_vllm_topk_skip_schedule() {
        let mut metadata = base_metadata(HfArchitectureKind::DeepSeekV32, 4);
        metadata.hidden_size = 7168;
        metadata.num_attention_heads = 128;
        metadata.num_key_value_heads = 128;
        metadata.head_dim = 192;
        metadata.intermediate_size = 18432;
        metadata.q_lora_rank = Some(1536);
        metadata.kv_lora_rank = Some(512);
        metadata.qk_nope_head_dim = Some(128);
        metadata.qk_rope_head_dim = Some(64);
        metadata.v_head_dim = Some(128);
        metadata.index_topk = Some(2048);
        metadata.index_topk_freq = Some(2);
        metadata.index_skip_topk_offset = Some(2);
        metadata.index_n_heads = Some(64);
        metadata.index_head_dim = Some(128);
        metadata.torch_dtype = Some(DType::BF16);

        let layers = descriptor_marker_layers(&metadata).unwrap();

        assert!(
            layers
                .iter()
                .all(|layer| layer.deepseek.unwrap().mode == CUDA_HF_DEEPSEEK_MODE_V32_MLA_INDEXER)
        );
        assert_eq!(
            layers
                .iter()
                .map(|layer| {
                    (layer.deepseek.unwrap().flags & CUDA_HF_DEEPSEEK_FLAG_SPARSE_INDEXER) != 0
                })
                .collect::<Vec<_>>(),
            vec![true, true, false, true]
        );
        assert_eq!(layers[2].deepseek.unwrap().index_topk, 2048);
    }

    fn base_metadata(architecture: HfArchitectureKind, layers: usize) -> HfModelMetadata {
        HfModelMetadata {
            architecture,
            hidden_size: 4,
            num_hidden_layers: layers,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 4,
            intermediate_size: 8,
            vocab_size: 16,
            max_position_embeddings: None,
            sliding_window: None,
            rope_theta: None,
            rope_scaling: None,
            compress_rope_theta: None,
            rms_norm_eps: Some(1e-5),
            bos_token_id: None,
            eos_token_id: None,
            tie_word_embeddings: false,
            hidden_act: Some("silu".to_string()),
            attention_bias: false,
            attention_qkv_bias: false,
            attention_output_bias: false,
            qk_norm: true,
            mlp_bias: false,
            linear_conv_kernel_dim: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            attention_layer_types: vec![HfAttentionLayerKind::Full; layers],
            mlp_layer_types: vec![HfMlpLayerKind::Dense; layers],
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            num_experts: None,
            num_experts_per_tok: None,
            decoder_sparse_step: None,
            norm_topk_prob: false,
            moe_first_k_dense_replace: None,
            moe_layer_freq: None,
            num_expert_groups: None,
            topk_group: None,
            topk_method: None,
            scoring_func: None,
            routed_scaling_factor: None,
            q_lora_rank: None,
            kv_lora_rank: None,
            o_lora_rank: None,
            o_groups: None,
            qk_nope_head_dim: None,
            qk_rope_head_dim: None,
            v_head_dim: None,
            index_topk: None,
            index_topk_freq: None,
            index_skip_topk_offset: None,
            index_topk_pattern: Vec::new(),
            index_n_heads: None,
            index_head_dim: None,
            compress_ratios: Vec::new(),
            hc_mult: None,
            hc_sinkhorn_iters: None,
            hc_eps: None,
            num_nextn_predict_layers: None,
            num_hash_layers: None,
            swiglu_limit: None,
            expert_dtype: None,
            torch_dtype: Some(DType::F16),
        }
    }
}
