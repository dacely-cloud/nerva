use nerva_core::types::error::Result;
use nerva_cuda::decode::hf_sequence::weight_plan::{
    CudaHfDecodeSequenceWeightBlock, hash_weight_blocks,
};
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::causal_lm::types::HfCausalLmLoaded;

use crate::engine::hf_cuda_decode::descriptors::cuda_weight_descriptors;
use crate::engine::hf_cuda_decode::summary::HfCudaResidentWeightSummary;
use crate::engine::runtime::Runtime;
use crate::residency::budget::ResidencyBudget;
use crate::weights::execution::plan::ResidentWeightExecutionPlan;
use crate::weights::execution::strategy::ResidentWeightExecutionStrategy;

pub(super) struct LoadedResidentWeightSummary {
    pub summary: HfCudaResidentWeightSummary,
    pub descriptors: Vec<CudaHfDecodeSequenceWeightBlock>,
}

pub(super) fn loaded_resident_weight_summary(
    runtime: &Runtime,
    loaded: &HfCausalLmLoaded,
    compute_capability: Option<u32>,
) -> Result<LoadedResidentWeightSummary> {
    let compute_capability = compute_capability.or_else(cuda_compute_capability);
    let manifest = &loaded.summary.manifest;
    let hotset_bytes = default_hotset_bytes(manifest.total_weight_bytes);
    let budget = ResidencyBudget::new(hotset_bytes, 0, manifest.total_weight_bytes);
    let mut table = runtime.materialize_hf_weight_manifest_with_budget(manifest, budget)?;
    let hotset = runtime.promote_resident_weight_hotset(&mut table, hotset_bytes)?;
    let plan = runtime.plan_resident_weight_execution(
        &table,
        loaded.summary.manifest.entries.len(),
        compute_capability,
    )?;
    let run = runtime.execute_resident_weight_execution_plan(&table, &plan)?;
    let resident_bytes = strategy_bytes(&plan, ResidentWeightExecutionStrategy::GpuResident);
    let staged_bytes = strategy_bytes(&plan, ResidentWeightExecutionStrategy::GpuStaged);
    let fallback_bytes = strategy_bytes(&plan, ResidentWeightExecutionStrategy::CpuExactFallback);
    let descriptors = cuda_weight_descriptors(loaded, &plan)?;
    let descriptor_hash = hash_weight_blocks(&descriptors);

    let summary = HfCudaResidentWeightSummary {
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
        cuda_contract_blocks: 0,
        cuda_contract_weight_bytes: 0,
        cuda_contract_descriptor_blocks: 0,
        cuda_contract_descriptor_hash: 0,
        cuda_contract_gpu_resident_h2d_bytes: 0,
        cuda_contract_gpu_staged_h2d_bytes: 0,
        cuda_contract_matched: false,
        hot_path_allocations: hotset.hot_path_allocations
            + run.hot_path_allocations
            + plan.ledger.hot_path_allocations,
    };
    Ok(LoadedResidentWeightSummary {
        summary,
        descriptors,
    })
}

pub(super) fn strategy_bytes(
    plan: &ResidentWeightExecutionPlan,
    strategy: ResidentWeightExecutionStrategy,
) -> u64 {
    plan.steps
        .iter()
        .filter(|step| step.strategy == strategy)
        .map(|step| step.bytes as u64)
        .sum()
}

pub(super) fn default_hotset_bytes(total_weight_bytes: usize) -> usize {
    total_weight_bytes
        .saturating_div(2)
        .max(1)
        .min(MAX_HOTSET_BYTES)
}

pub(super) fn default_large_file_backed_hotset_bytes(total_weight_bytes: usize) -> usize {
    resident_session_hotset_bytes(total_weight_bytes, cuda_free_memory_bytes())
}

pub(super) fn cuda_compute_capability() -> Option<u32> {
    let summary = nerva_cuda::smoke::probe::smoke();
    if summary.status != SmokeStatus::Ok {
        return None;
    }
    let major = u32::try_from(summary.compute_capability_major?).ok()?;
    let minor = u32::try_from(summary.compute_capability_minor?).ok()?;
    Some(major * 10 + minor)
}

const MAX_HOTSET_BYTES: usize = 512 * 1024 * 1024;
const FULL_RESIDENT_FREE_MEMORY_PERCENT: usize = 55;

fn resident_session_hotset_bytes(
    total_weight_bytes: usize,
    free_device_memory_bytes: Option<usize>,
) -> usize {
    let bounded = default_hotset_bytes(total_weight_bytes);
    if total_weight_bytes <= MAX_HOTSET_BYTES {
        return bounded;
    }
    let Some(free) = free_device_memory_bytes else {
        return bounded;
    };
    let full_resident_budget = free
        .saturating_mul(FULL_RESIDENT_FREE_MEMORY_PERCENT)
        .saturating_div(100);
    if total_weight_bytes <= full_resident_budget {
        total_weight_bytes
    } else {
        bounded
    }
}

fn cuda_free_memory_bytes() -> Option<usize> {
    let summary = nerva_cuda::smoke::probe::smoke();
    (summary.status == SmokeStatus::Ok)
        .then_some(summary.device_free_memory_bytes)
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::{MAX_HOTSET_BYTES, default_hotset_bytes, resident_session_hotset_bytes};

    #[test]
    fn large_file_backed_hotset_uses_full_residency_only_when_capacity_allows() {
        let large = MAX_HOTSET_BYTES * 4;

        assert_eq!(resident_session_hotset_bytes(large, Some(large * 2)), large);
        assert_eq!(
            resident_session_hotset_bytes(large, Some(large)),
            default_hotset_bytes(large)
        );
        assert_eq!(
            resident_session_hotset_bytes(large, None),
            default_hotset_bytes(large)
        );
    }

    #[test]
    fn small_file_backed_hotset_keeps_bounded_fixture_split() {
        assert_eq!(resident_session_hotset_bytes(100, Some(usize::MAX)), 50);
    }
}
