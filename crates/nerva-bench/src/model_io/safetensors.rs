use std::path::PathBuf;

use crate::model_io::config::load_manifest_from_config;

pub(crate) fn run_safetensors_probe(
    config_path: Option<String>,
    safetensors_path: Option<String>,
) -> Result<String, String> {
    match (config_path, safetensors_path) {
        (None, None) => nerva_model::weights::safetensors::probe::safetensors_header_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("safetensors header probe failed: {err:?}")),
        (Some(config_path), Some(safetensors_path)) => {
            let config = std::fs::read_to_string(&config_path)
                .map_err(|err| format!("failed to read {config_path}: {err}"))?;
            let manifest = load_manifest_from_config(&config)?;
            let header = nerva_model::weights::file::read_safetensors_header_file(PathBuf::from(
                &safetensors_path,
            ))
            .map_err(|err| format!("safetensors header read failed: {err:?}"))?;
            let validation = nerva_model::weights::safetensors::validation::validate_safetensors_header_for_manifest(
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

pub(crate) fn load_safetensors_shard_plan(
    config_path: Option<String>,
    index_path: Option<String>,
    checkpoint_dir: Option<String>,
) -> Result<nerva_model::weights::safetensors::shard::SafetensorsShardPlan, String> {
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
    let manifest = load_manifest_from_config(&config)?;
    let index_json = std::fs::read_to_string(&index_path)
        .map_err(|err| format!("failed to read {index_path}: {err}"))?;
    let shard_files =
        nerva_model::weights::safetensors::planner::required_safetensors_shards_for_manifest(
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
            nerva_model::weights::safetensors::shard::SafetensorsShardHeader::new(
                file_name,
                &header.header_json,
            )
        })
        .collect::<Vec<_>>();
    let shard_plan =
        nerva_model::weights::safetensors::planner::plan_safetensors_shards_for_manifest(
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
