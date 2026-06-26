use std::path::PathBuf;

use nerva_runtime::engine::residency::ResidencyBudget;
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

use crate::model_io::config::load_manifest_from_optional_config;
use crate::model_io::safetensors::load_safetensors_shard_plan;

pub(crate) fn run_resident_shard_probe(
    config_path: Option<String>,
    index_path: Option<String>,
    checkpoint_dir: Option<String>,
    max_task_bytes: usize,
) -> Result<String, String> {
    let checkpoint_dir_for_io = checkpoint_dir
        .as_ref()
        .map(PathBuf::from)
        .ok_or_else(|| {
            "sharded safetensors probes require config.json, model.safetensors.index.json, and checkpoint_dir"
                .to_string()
        })?;
    let shard_plan = load_safetensors_shard_plan(config_path, index_path, checkpoint_dir)?;
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let mut table = runtime
        .materialize_safetensors_shard_plan(&shard_plan)
        .map_err(|err| format!("resident shard materialization failed: {err:?}"))?;
    let prefetch = runtime
        .plan_resident_weight_prefetch(&table, max_task_bytes)
        .map_err(|err| format!("resident weight prefetch planning failed: {err:?}"))?;
    let execution = runtime
        .execute_resident_weight_prefetch_plan_from_files(
            &mut table,
            &prefetch,
            checkpoint_dir_for_io,
        )
        .map_err(|err| format!("resident weight file prefetch execution failed: {err:?}"))?;
    Ok(format!(
        "{{\"status\":\"ok\",\"blocks\":{},\"total_weight_bytes\":{},\"dram_used_bytes\":{},\"residency_decisions\":{},\"manifest_hash\":{},\"prefetch\":{},\"execution\":{}}}",
        table.entries.len(),
        table.total_weight_bytes,
        table
            .registry
            .used_bytes(nerva_core::types::memory::MemoryTier::Dram),
        table.ledger.residency_decisions.len(),
        table.manifest_hash,
        prefetch.to_json(),
        execution.to_json(),
    ))
}

pub(crate) fn run_resident_weight_probe(config_path: Option<String>) -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    match config_path {
        Some(path) => {
            let manifest = load_manifest_from_optional_config(Some(path))?;
            let table = runtime
                .materialize_hf_weight_manifest(&manifest)
                .map_err(|err| format!("resident weight materialization failed: {err:?}"))?;
            Ok(format!(
                "{{\"status\":\"ok\",\"blocks\":{},\"total_weight_bytes\":{},\"dram_used_bytes\":{},\"manifest_hash\":{},\"hot_path_allocations\":{}}}",
                table.entries.len(),
                table.total_weight_bytes,
                table
                    .registry
                    .used_bytes(nerva_core::types::memory::MemoryTier::Dram),
                table.manifest_hash,
                table.ledger.hot_path_allocations,
            ))
        }
        None => runtime
            .run_resident_weight_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("resident weight probe failed: {err:?}")),
    }
}

pub(crate) fn run_hotset_probe(
    config_path: Option<String>,
    vram_bytes: usize,
    max_promote_bytes: usize,
) -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let manifest = load_manifest_from_optional_config(config_path)?;
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(vram_bytes, 0, manifest.total_weight_bytes),
        )
        .map_err(|err| format!("resident weight materialization failed: {err:?}"))?;
    let hotset = runtime
        .promote_resident_weight_hotset(&mut table, max_promote_bytes)
        .map_err(|err| format!("resident weight hotset promotion failed: {err:?}"))?;
    Ok(format!(
        "{{\"status\":\"ok\",\"blocks\":{},\"total_weight_bytes\":{},\"manifest_hash\":{},\"hotset\":{}}}",
        table.entries.len(),
        table.total_weight_bytes,
        table.manifest_hash,
        hotset.to_json(),
    ))
}

pub(crate) fn run_weight_execution_probe(
    config_path: Option<String>,
    vram_bytes: usize,
    max_promote_bytes: usize,
    max_steps: usize,
    compute_capability: Option<u32>,
) -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let manifest = load_manifest_from_optional_config(config_path)?;
    let mut table = runtime
        .materialize_hf_weight_manifest_with_budget(
            &manifest,
            ResidencyBudget::new(vram_bytes, 0, manifest.total_weight_bytes),
        )
        .map_err(|err| format!("resident weight materialization failed: {err:?}"))?;
    let hotset = runtime
        .promote_resident_weight_hotset(&mut table, max_promote_bytes)
        .map_err(|err| format!("resident weight hotset promotion failed: {err:?}"))?;
    let execution = runtime
        .plan_resident_weight_execution(&table, max_steps, compute_capability)
        .map_err(|err| format!("resident weight execution planning failed: {err:?}"))?;
    let execution_run = runtime
        .execute_resident_weight_execution_plan(&table, &execution)
        .map_err(|err| format!("resident weight execution run failed: {err:?}"))?;
    Ok(format!(
        "{{\"status\":\"ok\",\"blocks\":{},\"total_weight_bytes\":{},\"manifest_hash\":{},\"hotset\":{},\"execution\":{},\"run\":{}}}",
        table.entries.len(),
        table.total_weight_bytes,
        table.manifest_hash,
        hotset.to_json(),
        execution.to_json(),
        execution_run.to_json(),
    ))
}
