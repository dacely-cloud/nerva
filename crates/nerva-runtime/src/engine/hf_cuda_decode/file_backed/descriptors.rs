use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;

use nerva_core::types::error::{NervaError, Result};
use nerva_cuda::decode::hf_chain::layer::CudaHfDecodeChainLayer;
use nerva_cuda::decode::hf_sequence::weight_plan::{
    CudaHfDecodeSequenceWeightBlock, hash_weight_blocks,
};
use nerva_model::hf::metadata::HfModelMetadata;

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
}

pub(super) fn descriptor_marker_layers(
    metadata: &HfModelMetadata,
) -> Vec<CudaHfDecodeChainLayer<'static>> {
    let qk_norm = metadata.qk_norm.then_some(&MARKER[..]);
    let attn_bias = metadata.attention_bias.then_some(&MARKER[..]);
    (0..metadata.num_hidden_layers)
        .map(|_| CudaHfDecodeChainLayer {
            rms_attn_weight: &EMPTY,
            rms_mlp_weight: &EMPTY,
            w_q: &EMPTY,
            w_k: &EMPTY,
            q_norm_weight: qk_norm,
            k_norm_weight: qk_norm,
            w_v: &EMPTY,
            w_o: &EMPTY,
            q_bias: attn_bias,
            k_bias: attn_bias,
            v_bias: attn_bias,
            o_bias: attn_bias,
            w_gate: &EMPTY,
            w_up: &EMPTY,
            w_down: &EMPTY,
        })
        .collect()
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
    } = cuda_weight_descriptors(weights, &plan)?;
    let descriptor_hash = hash_weight_blocks(&descriptors);
    let resident_bytes = strategy_bytes(&plan, ResidentWeightExecutionStrategy::GpuResident);
    let staged_bytes = strategy_bytes(&plan, ResidentWeightExecutionStrategy::GpuStaged);
    let fallback_bytes = strategy_bytes(&plan, ResidentWeightExecutionStrategy::CpuExactFallback);
    Ok(ShardBackedResidentWeights {
        summary: HfCudaResidentWeightSummary {
            plan_steps: plan.steps.len() as u64,
            plan_weight_bytes: plan.total_weight_bytes as u64,
            plan_descriptor_blocks: descriptors.len() as u64,
            plan_descriptor_hash: descriptor_hash,
            hotset_promoted_blocks: hotset.promoted_blocks as u64,
            hotset_promoted_bytes: hotset.promoted_bytes as u64,
            hotset_kept_dram_blocks: hotset.kept_dram_blocks as u64,
            plan_gpu_resident_weight_bytes: resident_bytes,
            plan_gpu_staged_weight_bytes: staged_bytes,
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
    })
}

struct DescriptorTable {
    descriptors: Vec<CudaHfDecodeSequenceWeightBlock>,
    source_paths: Vec<CString>,
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
        let source_path = CString::new(source_path.as_os_str().as_bytes()).map_err(|_| {
            NervaError::InvalidArgument {
                reason: format!(
                    "safetensors shard path for {} contains a nul byte",
                    shard.tensor_name
                ),
            }
        })?;
        descriptors.push(CudaHfDecodeSequenceWeightBlock {
            host_source: std::ptr::null(),
            source_file: source_path.as_ptr(),
            source_file_len: source_path.as_bytes().len() as u64,
            file_offset_begin: shard.file_offset_begin as u64,
            block_id: step.block_id.0,
            block_version: step.block_version,
            offset_bytes,
            bytes: step.bytes as u64,
            strategy: cuda_weight_strategy(step.strategy)?,
            reserved: 0,
        });
        source_paths.push(source_path);
        offset_bytes = offset_bytes.checked_add(step.bytes as u64).ok_or_else(|| {
            NervaError::AllocationFailed {
                bytes: step.bytes,
                reason: "CUDA shard-backed descriptor offset overflow".to_string(),
            }
        })?;
    }
    Ok(DescriptorTable {
        descriptors,
        source_paths,
    })
}
