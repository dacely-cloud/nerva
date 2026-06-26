use std::fs;

use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

use crate::json::json_escape;

pub(crate) fn safetensors_file_header_acceptance() -> Result<(bool, String), String> {
    let config = r#"{
            "model_type": "llama",
            "hidden_size": 2,
            "intermediate_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 4,
            "torch_dtype": "float16"
        }"#;
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(config)
        .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
    let layout = nerva_model::weights::layout::plan_hf_weight_layout(&metadata)
        .map_err(|err| format!("HF layout probe failed: {err:?}"))?;
    let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&layout)
        .map_err(|err| format!("HF manifest probe failed: {err:?}"))?;
    let header =
        nerva_model::weights::safetensors::header::synthetic_safetensors_header_for_manifest(
            &manifest,
        )
        .map_err(|err| format!("safetensors header generation failed: {err:?}"))?;
    let dir = std::env::temp_dir().join(format!(
        "nerva-acceptance-safetensors-{}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
    let path = dir.join("model.safetensors");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(8 + header.len() + manifest.total_weight_bytes, 0);
    fs::write(&path, bytes).map_err(|err| format!("failed to write {}: {err}", path.display()))?;

    let file_header = nerva_model::weights::file::read_safetensors_header_file(&path)
        .map_err(|err| format!("safetensors header read failed: {err:?}"))?;
    let validation =
        nerva_model::weights::safetensors::validation::validate_safetensors_header_for_manifest(
            &file_header.header_json,
            &manifest,
        )
        .map_err(|err| format!("safetensors manifest validation failed: {err:?}"))?;

    let _ = fs::remove_file(&path);
    let _ = fs::remove_dir(&dir);

    let passed = file_header.header_bytes == header.len()
        && file_header.data_start == 8 + header.len()
        && file_header.payload_bytes == manifest.total_weight_bytes as u64
        && validation.validated_tensors == manifest.entries.len()
        && validation.total_data_bytes == manifest.total_weight_bytes
        && validation.header_hash != 0;
    Ok((
        passed,
        format!(
            "manifest_entries={} validated_tensors={} header_bytes={} data_start={} payload_bytes={} total_data_bytes={} header_hash={}",
            manifest.entries.len(),
            validation.validated_tensors,
            file_header.header_bytes,
            file_header.data_start,
            file_header.payload_bytes,
            validation.total_data_bytes,
            validation.header_hash,
        ),
    ))
}

pub(crate) fn safetensors_file_prefetch_acceptance() -> Result<(bool, String), String> {
    let config = r#"{
            "model_type": "llama",
            "hidden_size": 2,
            "intermediate_size": 4,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 4,
            "torch_dtype": "float16"
        }"#;
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(config)
        .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
    let layout = nerva_model::weights::layout::plan_hf_weight_layout(&metadata)
        .map_err(|err| format!("HF layout probe failed: {err:?}"))?;
    let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&layout)
        .map_err(|err| format!("HF manifest probe failed: {err:?}"))?;
    let header =
        nerva_model::weights::safetensors::header::synthetic_safetensors_header_for_manifest(
            &manifest,
        )
        .map_err(|err| format!("safetensors header generation failed: {err:?}"))?;
    let shard_name = "model.safetensors";
    let index = single_shard_index_json(&manifest, shard_name);
    let shard_plan =
        nerva_model::weights::safetensors::planner::plan_safetensors_shards_for_manifest(
            &index,
            &[
                nerva_model::weights::safetensors::shard::SafetensorsShardHeader::new(
                    shard_name, &header,
                ),
            ],
            &manifest,
        )
        .map_err(|err| format!("safetensors shard planning failed: {err:?}"))?;

    let dir =
        std::env::temp_dir().join(format!("nerva-acceptance-prefetch-{}", std::process::id()));
    fs::create_dir_all(&dir).map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
    let path = dir.join(shard_name);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for index in 0..manifest.total_weight_bytes {
        bytes.push(((index * 17 + 11) % 251) as u8);
    }
    fs::write(&path, bytes).map_err(|err| format!("failed to write {}: {err}", path.display()))?;

    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let mut table = runtime
        .materialize_safetensors_shard_plan(&shard_plan)
        .map_err(|err| format!("resident shard materialization failed: {err:?}"))?;
    let prefetch = runtime
        .plan_resident_weight_prefetch(&table, 32)
        .map_err(|err| format!("resident prefetch planning failed: {err:?}"))?;
    let execution = runtime
        .execute_resident_weight_prefetch_plan_from_files(&mut table, &prefetch, &dir)
        .map_err(|err| format!("resident file prefetch execution failed: {err:?}"))?;

    let _ = fs::remove_file(&path);
    let _ = fs::remove_dir(&dir);

    let passed = execution.tasks == prefetch.tasks.len()
        && execution.completed_blocks == table.entries.len()
        && execution.total_bytes == manifest.total_weight_bytes
        && execution.disk_read_events == prefetch.tasks.len() as u64
        && execution.copy_events == prefetch.tasks.len() as u64
        && execution.ready_blocks == table.entries.len()
        && execution.data_hash != 0
        && execution.hot_path_allocations == 0;
    Ok((
        passed,
        format!(
            "entries={} tasks={} completed_blocks={} total_bytes={} disk_read_events={} copy_events={} ready_blocks={} data_hash={} hot_path_allocations={}",
            table.entries.len(),
            execution.tasks,
            execution.completed_blocks,
            execution.total_bytes,
            execution.disk_read_events,
            execution.copy_events,
            execution.ready_blocks,
            execution.data_hash,
            execution.hot_path_allocations,
        ),
    ))
}

fn single_shard_index_json(
    manifest: &nerva_model::weights::manifest::HfTensorManifest,
    shard_name: &str,
) -> String {
    let mut out = format!(
        "{{\"metadata\":{{\"total_size\":{}}},\"weight_map\":{{",
        manifest.total_weight_bytes
    );
    for (index, entry) in manifest.entries.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(&entry.name));
        out.push_str("\":\"");
        out.push_str(&json_escape(shard_name));
        out.push('"');
    }
    out.push_str("}}");
    out
}
