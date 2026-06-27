use nerva_core::types::dtype::DType;

pub(crate) fn model_manifest_acceptance() -> Result<(bool, String), String> {
    let metadata = nerva_model::hf::probe::hf_metadata_probe()
        .map_err(|err| format!("HF metadata probe failed: {err:?}"))?;
    let layout = nerva_model::weights::layout::probe::hf_weight_layout_probe()
        .map_err(|err| format!("HF layout probe failed: {err:?}"))?;
    let manifest = nerva_model::weights::manifest::hf_tensor_manifest_probe()
        .map_err(|err| format!("HF manifest probe failed: {err:?}"))?;
    let safetensors = nerva_model::weights::safetensors::probe::safetensors_header_probe()
        .map_err(|err| format!("safetensors header probe failed: {err:?}"))?;

    let metadata_body = &metadata.metadata;
    let expected_static_blocks = if metadata_body.tie_word_embeddings {
        2
    } else {
        3
    };
    let expected_blocks = metadata_body
        .num_hidden_layers
        .checked_mul(9)
        .and_then(|layer_blocks| layer_blocks.checked_add(expected_static_blocks))
        .ok_or_else(|| "expected HF block count overflowed".to_string())?;
    let metadata_passed = metadata_body.architecture.as_str() == "llama"
        && metadata_body.hidden_size == 4096
        && metadata_body.num_hidden_layers == 32
        && metadata_body.num_attention_heads == 32
        && metadata_body.num_key_value_heads == 8
        && metadata_body.torch_dtype == Some(DType::BF16)
        && metadata.metadata_hash != 0;
    let layout_passed = layout.plan.metadata == *metadata_body
        && layout.plan.blocks.len() == expected_blocks
        && layout.plan.total_weight_bytes > 0
        && layout.layout_hash != 0;
    let manifest_passed = manifest.manifest.entries.len() == layout.plan.blocks.len()
        && manifest.manifest.total_weight_bytes == layout.plan.total_weight_bytes
        && manifest.manifest.manifest_hash != 0;
    let validation = &safetensors.validation;
    let safetensors_passed = validation.manifest_entries == manifest.manifest.entries.len()
        && validation.validated_tensors == manifest.manifest.entries.len()
        && validation.total_data_bytes == manifest.manifest.total_weight_bytes
        && validation.manifest_hash == manifest.manifest.manifest_hash
        && validation.header_bytes > 0
        && validation.header_hash != 0;

    Ok((
        metadata_passed && layout_passed && manifest_passed && safetensors_passed,
        format!(
            "architecture={} layers={} hidden={} kv_heads={} dtype={:?} expected_blocks={} layout_blocks={} manifest_entries={} safetensors_validated={} total_weight_bytes={} metadata_hash={} layout_hash={} manifest_hash={} header_hash={}",
            metadata_body.architecture.as_str(),
            metadata_body.num_hidden_layers,
            metadata_body.hidden_size,
            metadata_body.num_key_value_heads,
            metadata_body.torch_dtype,
            expected_blocks,
            layout.plan.blocks.len(),
            manifest.manifest.entries.len(),
            validation.validated_tensors,
            manifest.manifest.total_weight_bytes,
            metadata.metadata_hash,
            layout.layout_hash,
            manifest.manifest.manifest_hash,
            validation.header_hash,
        ),
    ))
}
