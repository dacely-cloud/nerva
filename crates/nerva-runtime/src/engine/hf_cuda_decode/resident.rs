use nerva_core::types::error::Result;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_model::causal_lm::types::HfCausalLmLoaded;

use crate::engine::hf_cuda_decode::summary::HfCudaResidentWeightSummary;
use crate::engine::runtime::Runtime;
use crate::residency::budget::ResidencyBudget;

pub(super) fn loaded_resident_weight_summary(
    runtime: &Runtime,
    loaded: &HfCausalLmLoaded,
    compute_capability: Option<u32>,
) -> Result<HfCudaResidentWeightSummary> {
    let compute_capability = compute_capability.or_else(cuda_compute_capability);
    let manifest = &loaded.summary.manifest;
    let budget = ResidencyBudget::new(0, 0, manifest.total_weight_bytes);
    let table = runtime.materialize_hf_weight_manifest_with_budget(manifest, budget)?;
    let plan = runtime.plan_resident_weight_execution(
        &table,
        loaded.summary.manifest.entries.len(),
        compute_capability,
    )?;
    let run = runtime.execute_resident_weight_execution_plan(&table, &plan)?;

    Ok(HfCudaResidentWeightSummary {
        plan_steps: plan.steps.len() as u64,
        plan_weight_bytes: plan.total_weight_bytes as u64,
        plan_gpu_resident_steps: plan.gpu_resident_steps,
        plan_gpu_staged_steps: plan.gpu_staged_steps,
        plan_fallback_steps: plan.fallback_steps,
        plan_block_version_dependencies: plan.block_version_dependencies,
        run_steps: run.steps as u64,
        run_gpu_resident_steps: run.gpu_resident_steps,
        run_gpu_staged_steps: run.gpu_staged_steps,
        run_fallback_steps: run.fallback_steps,
        run_block_version_dependencies: run.block_version_dependencies,
        hot_path_allocations: run.hot_path_allocations + plan.ledger.hot_path_allocations,
    })
}

fn cuda_compute_capability() -> Option<u32> {
    let summary = nerva_cuda::smoke::probe::smoke();
    if summary.status != SmokeStatus::Ok {
        return None;
    }
    let major = u32::try_from(summary.compute_capability_major?).ok()?;
    let minor = u32::try_from(summary.compute_capability_minor?).ok()?;
    Some(major * 10 + minor)
}
