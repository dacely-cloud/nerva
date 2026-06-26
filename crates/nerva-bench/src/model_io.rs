use std::path::PathBuf;

use nerva_runtime::engine::{ResidencyBudget, Runtime, RuntimeConfig};

pub(crate) fn run_metadata_probe(config_path: Option<String>) -> Result<String, String> {
    match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            let metadata = nerva_model::hf::parser::parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            Ok(metadata.to_json())
        }
        None => nerva_model::hf::probe::hf_metadata_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("HF metadata probe failed: {err:?}")),
    }
}

pub(crate) fn run_layout_probe(config_path: Option<String>) -> Result<String, String> {
    match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            let metadata = nerva_model::hf::parser::parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            let plan = nerva_model::weights::layout::plan_hf_weight_layout(&metadata)
                .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
            Ok(plan.to_json())
        }
        None => nerva_model::weights::layout::hf_weight_layout_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("HF weight layout probe failed: {err:?}")),
    }
}

pub(crate) fn run_manifest_probe(config_path: Option<String>) -> Result<String, String> {
    match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            let metadata = nerva_model::hf::parser::parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            let plan = nerva_model::weights::layout::plan_hf_weight_layout(&metadata)
                .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
            let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&plan)
                .map_err(|err| format!("HF tensor manifest failed: {err:?}"))?;
            Ok(manifest.to_json())
        }
        None => nerva_model::weights::manifest::hf_tensor_manifest_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("HF tensor manifest probe failed: {err:?}")),
    }
}

pub(crate) fn run_safetensors_probe(
    config_path: Option<String>,
    safetensors_path: Option<String>,
) -> Result<String, String> {
    match (config_path, safetensors_path) {
        (None, None) => nerva_model::weights::safetensors::safetensors_header_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("safetensors header probe failed: {err:?}")),
        (Some(config_path), Some(safetensors_path)) => {
            let config = std::fs::read_to_string(&config_path)
                .map_err(|err| format!("failed to read {config_path}: {err}"))?;
            let metadata = nerva_model::hf::parser::parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            let plan = nerva_model::weights::layout::plan_hf_weight_layout(&metadata)
                .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
            let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&plan)
                .map_err(|err| format!("HF tensor manifest failed: {err:?}"))?;
            let header = nerva_model::weights::file::read_safetensors_header_file(PathBuf::from(
                &safetensors_path,
            ))
            .map_err(|err| format!("safetensors header read failed: {err:?}"))?;
            let validation =
                nerva_model::weights::safetensors::validate_safetensors_header_for_manifest(
                    &header.header_json,
                    &manifest,
                )
                .map_err(|err| format!("safetensors manifest validation failed: {err:?}"))?;
            header
                .require_payload_bytes(validation.total_data_bytes)
                .map_err(|err| format!("safetensors payload validation failed: {err:?}"))?;
            Ok(format!(
                "{{\"status\":\"ok\",\"file_header\":{},\"validation\":{}}}",
                header.to_json(),
                validation.to_json(),
            ))
        }
        _ => Err(
            "safetensors requires either no args or both config.json and model.safetensors"
                .to_string(),
        ),
    }
}

pub(crate) fn run_safetensors_shard_probe(
    config_path: Option<String>,
    index_path: Option<String>,
    checkpoint_dir: Option<String>,
) -> Result<String, String> {
    let shard_plan = load_safetensors_shard_plan(config_path, index_path, checkpoint_dir)?;
    Ok(format!(
        "{{\"status\":\"ok\",\"plan\":{}}}",
        shard_plan.to_json()
    ))
}

pub(crate) fn run_resident_shard_probe(
    config_path: Option<String>,
    index_path: Option<String>,
    checkpoint_dir: Option<String>,
    max_task_bytes: usize,
) -> Result<String, String> {
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
        .execute_resident_weight_prefetch_plan(&mut table, &prefetch)
        .map_err(|err| format!("resident weight prefetch execution failed: {err:?}"))?;
    Ok(format!(
        "{{\"status\":\"ok\",\"blocks\":{},\"total_weight_bytes\":{},\"dram_used_bytes\":{},\"residency_decisions\":{},\"manifest_hash\":{},\"prefetch\":{},\"execution\":{}}}",
        table.entries.len(),
        table.total_weight_bytes,
        table
            .registry
            .used_bytes(nerva_core::types::MemoryTier::Dram),
        table.ledger.residency_decisions.len(),
        table.manifest_hash,
        prefetch.to_json(),
        execution.to_json(),
    ))
}

fn load_safetensors_shard_plan(
    config_path: Option<String>,
    index_path: Option<String>,
    checkpoint_dir: Option<String>,
) -> Result<nerva_model::weights::safetensors::SafetensorsShardPlan, String> {
    let (Some(config_path), Some(index_path), Some(checkpoint_dir)) =
        (config_path, index_path, checkpoint_dir)
    else {
        return Err(
            "sharded safetensors probes require config.json, model.safetensors.index.json, and checkpoint_dir"
                .to_string(),
        );
    };
    let config = std::fs::read_to_string(&config_path)
        .map_err(|err| format!("failed to read {config_path}: {err}"))?;
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(&config)
        .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
    let plan = nerva_model::weights::layout::plan_hf_weight_layout(&metadata)
        .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
    let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&plan)
        .map_err(|err| format!("HF tensor manifest failed: {err:?}"))?;
    let index_json = std::fs::read_to_string(&index_path)
        .map_err(|err| format!("failed to read {index_path}: {err}"))?;
    let shard_files = nerva_model::weights::safetensors::required_safetensors_shards_for_manifest(
        &index_json,
        &manifest,
    )
    .map_err(|err| format!("safetensors index validation failed: {err:?}"))?;
    let checkpoint_dir = PathBuf::from(checkpoint_dir);
    let mut shard_headers = Vec::with_capacity(shard_files.len());
    for shard_file in shard_files {
        let header = nerva_model::weights::file::read_safetensors_header_file(
            checkpoint_dir.join(&shard_file),
        )
        .map_err(|err| format!("safetensors header read failed: {err:?}"))?;
        shard_headers.push((shard_file, header));
    }
    let shard_header_refs = shard_headers
        .iter()
        .map(|(file_name, header)| {
            nerva_model::weights::safetensors::SafetensorsShardHeader::new(
                file_name,
                &header.header_json,
            )
        })
        .collect::<Vec<_>>();
    let shard_plan = nerva_model::weights::safetensors::plan_safetensors_shards_for_manifest(
        &index_json,
        &shard_header_refs,
        &manifest,
    )
    .map_err(|err| format!("safetensors shard plan failed: {err:?}"))?;
    for entry in &shard_plan.entries {
        let (_, header) = shard_headers
            .iter()
            .find(|(file_name, _)| file_name == &entry.shard_file)
            .ok_or_else(|| format!("missing loaded header for shard {}", entry.shard_file))?;
        header
            .require_file_offset_end(entry.file_offset_end)
            .map_err(|err| format!("safetensors shard payload validation failed: {err:?}"))?;
    }
    Ok(shard_plan)
}

pub(crate) fn run_resident_weight_probe(config_path: Option<String>) -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            let metadata = nerva_model::hf::parser::parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            let plan = nerva_model::weights::layout::plan_hf_weight_layout(&metadata)
                .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
            let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&plan)
                .map_err(|err| format!("HF tensor manifest failed: {err:?}"))?;
            let table = runtime
                .materialize_hf_weight_manifest(&manifest)
                .map_err(|err| format!("resident weight materialization failed: {err:?}"))?;
            Ok(format!(
                "{{\"status\":\"ok\",\"blocks\":{},\"total_weight_bytes\":{},\"dram_used_bytes\":{},\"manifest_hash\":{},\"hot_path_allocations\":{}}}",
                table.entries.len(),
                table.total_weight_bytes,
                table
                    .registry
                    .used_bytes(nerva_core::types::MemoryTier::Dram),
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

fn load_manifest_from_optional_config(
    config_path: Option<String>,
) -> Result<nerva_model::weights::manifest::HfTensorManifest, String> {
    match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            let metadata = nerva_model::hf::parser::parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            let plan = nerva_model::weights::layout::plan_hf_weight_layout(&metadata)
                .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
            nerva_model::weights::manifest::build_hf_tensor_manifest(&plan)
                .map_err(|err| format!("HF tensor manifest failed: {err:?}"))
        }
        None => nerva_model::weights::manifest::hf_tensor_manifest_probe()
            .map(|summary| summary.manifest)
            .map_err(|err| format!("HF tensor manifest probe failed: {err:?}")),
    }
}
