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
            let plan = nerva_model::weights::layout::plan::plan_hf_weight_layout(&metadata)
                .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
            Ok(plan.to_json())
        }
        None => nerva_model::weights::layout::probe::hf_weight_layout_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("HF weight layout probe failed: {err:?}")),
    }
}

pub(crate) fn run_manifest_probe(config_path: Option<String>) -> Result<String, String> {
    match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            let manifest = load_manifest_from_config(&config)?;
            Ok(manifest.to_json())
        }
        None => nerva_model::weights::manifest::hf_tensor_manifest_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("HF tensor manifest probe failed: {err:?}")),
    }
}

pub(crate) fn load_manifest_from_optional_config(
    config_path: Option<String>,
) -> Result<nerva_model::weights::manifest::HfTensorManifest, String> {
    match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            load_manifest_from_config(&config)
        }
        None => nerva_model::weights::manifest::hf_tensor_manifest_probe()
            .map(|summary| summary.manifest)
            .map_err(|err| format!("HF tensor manifest probe failed: {err:?}")),
    }
}

pub(crate) fn load_manifest_from_config(
    config: &str,
) -> Result<nerva_model::weights::manifest::HfTensorManifest, String> {
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(config)
        .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
    let plan = nerva_model::weights::layout::plan::plan_hf_weight_layout(&metadata)
        .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
    nerva_model::weights::manifest::build_hf_tensor_manifest(&plan)
        .map_err(|err| format!("HF tensor manifest failed: {err:?}"))
}
