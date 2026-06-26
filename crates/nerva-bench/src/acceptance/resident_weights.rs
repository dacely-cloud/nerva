use nerva_runtime::engine::residency::ResidencyBudget;
use nerva_runtime::engine::runtime::Runtime;

pub(crate) fn resident_weight_execution_acceptance(
    runtime: &Runtime,
) -> Result<(bool, String), String> {
    let manifest = nerva_model::weights::manifest::hf_tensor_manifest_probe()
        .map_err(|err| format!("HF tensor manifest probe failed: {err:?}"))?
        .manifest;
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(512 * 1024 * 1024, 0, manifest.total_weight_bytes),
        )
        .map_err(|err| format!("resident weight materialization failed: {err:?}"))?;
    let hotset = runtime
        .promote_resident_weight_hotset(&mut table, 512 * 1024 * 1024)
        .map_err(|err| format!("resident weight hotset promotion failed: {err:?}"))?;
    let plan = runtime
        .plan_resident_weight_execution(&table, 32, Some(89))
        .map_err(|err| format!("resident weight execution planning failed: {err:?}"))?;
    let run = runtime
        .execute_resident_weight_execution_plan(&table, &plan)
        .map_err(|err| format!("resident weight execution run failed: {err:?}"))?;

    let passed = hotset.promoted_blocks > 0
        && hotset.considered_blocks == manifest.entries.len()
        && hotset.kept_dram_blocks > 0
        && hotset.budget_limited_blocks > 0
        && hotset.residency_decisions == hotset.considered_blocks as u64
        && hotset.hot_path_allocations == 0
        && !plan.steps.is_empty()
        && plan.gpu_resident_steps > 0
        && plan.gpu_staged_steps > 0
        && plan.block_version_dependencies == plan.steps.len() as u64
        && plan.ledger.hot_path_allocations == 0
        && run.steps == plan.steps.len()
        && run.gpu_resident_steps == plan.gpu_resident_steps
        && run.gpu_staged_steps == plan.gpu_staged_steps
        && run.block_version_dependencies == run.steps as u64
        && run.hot_path_allocations == 0;

    Ok((
        passed,
        format!(
            "hotset_considered={} promoted_blocks={} kept_dram_blocks={} budget_limited_blocks={} capacity_limited_blocks={} hotset_decisions={} plan_steps={} plan_gpu_resident={} plan_gpu_staged={} plan_fallbacks={} plan_block_versions={} run_steps={} run_gpu_resident={} run_gpu_staged={} run_fallbacks={} run_block_versions={} hot_path_allocations={}",
            hotset.considered_blocks,
            hotset.promoted_blocks,
            hotset.kept_dram_blocks,
            hotset.budget_limited_blocks,
            hotset.capacity_limited_blocks,
            hotset.residency_decisions,
            plan.steps.len(),
            plan.gpu_resident_steps,
            plan.gpu_staged_steps,
            plan.fallback_decisions,
            plan.block_version_dependencies,
            run.steps,
            run.gpu_resident_steps,
            run.gpu_staged_steps,
            run.fallback_decisions,
            run.block_version_dependencies,
            hotset.hot_path_allocations
                + plan.ledger.hot_path_allocations
                + run.hot_path_allocations,
        ),
    ))
}
