use std::path::{Path, PathBuf};

use crate::acceptance::report::AcceptanceReport;
use crate::artifact::run::run_artifact;

pub(crate) fn push_artifact_reproducibility(report: &mut AcceptanceReport) {
    match run_artifact(
        Some("synthetic".to_string()),
        vec!["2".to_string(), "4".to_string()],
    ) {
        Ok(artifact) => report.push(
            "benchmark_artifact_reproducibility",
            artifact.contains("\"artifact_schema\":\"nerva-bench-v1\"")
                && artifact.contains("\"metadata\"")
                && artifact.contains("\"summary\"")
                && artifact.contains("\"command\":\"synthetic\"")
                && artifact.contains("\"args\":[\"2\",\"4\"]")
                && artifact.contains("\"command_line\":[\"cargo\",\"run\",\"-p\",\"nerva-bench\"")
                && artifact.contains("\"cwd\"")
                && artifact.contains("\"git_commit\"")
                && artifact.contains("\"package_version\"")
                && artifact.contains("\"profile\"")
                && artifact.contains("\"target\"")
                && artifact.contains("\"rustc_version\"")
                && artifact.contains("\"cargo_version\"")
                && artifact.contains("\"environment\"")
                && artifact.contains("\"CUDA_VISIBLE_DEVICES\"")
                && artifact.contains("\"HIP_VISIBLE_DEVICES\"")
                && artifact.contains("\"capabilities\"")
                && artifact.contains("\"kernel_release\"")
                && artifact.contains("\"topology\"")
                && artifact.contains("\"observed_token_hash\"")
                && artifact.contains("\"device_timeline_idle_ns\":0"),
            format!(
                "schema={} command={} args={} command_line={} environment={} capabilities={} summary_hash={} idle_zero={}",
                artifact.contains("\"artifact_schema\":\"nerva-bench-v1\""),
                artifact.contains("\"command\":\"synthetic\""),
                artifact.contains("\"args\":[\"2\",\"4\"]"),
                artifact.contains("\"command_line\""),
                artifact.contains("\"environment\""),
                artifact.contains("\"capabilities\""),
                artifact.contains("\"observed_token_hash\""),
                artifact.contains("\"device_timeline_idle_ns\":0"),
            ),
        ),
        Err(err) => report.push(
            "benchmark_artifact_reproducibility",
            false,
            format!("artifact reproducibility probe failed: {err}"),
        ),
    }
}

pub(crate) fn push_hf_checkpoint_artifacts(report: &mut AcceptanceReport) {
    match hf_checkpoint_artifact_result() {
        Ok((passed, details)) => report.push("hf_checkpoint_artifacts", passed, details),
        Err(err) => report.push("hf_checkpoint_artifacts", false, err),
    }
}

fn hf_checkpoint_artifact_result() -> Result<(bool, String), String> {
    let dir = write_tiny_hf_checkpoint_dir("nerva-acceptance-hf-artifact")?;
    let path = dir.to_string_lossy().into_owned();
    let cpu = run_artifact(
        Some("hf-decode".to_string()),
        vec![path.clone(), "ids:0,1".to_string(), "2".to_string()],
    );
    let cuda = run_artifact(
        Some("hf-cuda-decode".to_string()),
        vec![path, "ids:0,1".to_string(), "2".to_string()],
    );
    remove_tiny_hf_checkpoint_dir(&dir);
    let cpu = cpu?;
    let cuda = cuda?;
    Ok((
        artifact_has_hf_cpu_fields(&cpu) && artifact_has_hf_cuda_fields(&cuda),
        format!(
            "cpu_schema={} cpu_command={} cpu_summary={} cuda_schema={} cuda_command={} cuda_summary={} cuda_contract={} cuda_host_causality_zero={} cuda_hot_path_zero={} cuda_token_ledgers={} cuda_device_timeline={}",
            cpu.contains("\"artifact_schema\":\"nerva-bench-v1\""),
            cpu.contains("\"command\":\"hf-decode\""),
            cpu.contains("\"context_mode\":\"prompt_prefill_kv_decode\""),
            cuda.contains("\"artifact_schema\":\"nerva-bench-v1\""),
            cuda.contains("\"command\":\"hf-cuda-decode\""),
            cuda.contains("\"backend\":\"cuda\""),
            cuda.contains("\"cuda_contract_matched\":true"),
            cuda.contains("\"host_causality_edges\":0"),
            cuda.contains("\"hot_path_allocations\":0"),
            cuda.contains("\"token_ledgers\":["),
            cuda.contains("\"hf_cuda_sequence_device_timeline\""),
        ),
    ))
}

fn artifact_has_hf_cpu_fields(artifact: &str) -> bool {
    artifact.contains("\"artifact_schema\":\"nerva-bench-v1\"")
        && artifact.contains("\"command\":\"hf-decode\"")
        && artifact.contains("\"capabilities\"")
        && artifact.contains("\"summary\"")
        && artifact.contains("\"input_mode\":\"token_ids\"")
        && artifact.contains("\"context_mode\":\"prompt_prefill_kv_decode\"")
        && artifact.contains("\"prompt_token_ids\":[0,1]")
        && artifact.contains("\"manifest_entries\":12")
        && artifact.contains("\"hot_path_allocations\":0")
}

fn artifact_has_hf_cuda_fields(artifact: &str) -> bool {
    artifact.contains("\"artifact_schema\":\"nerva-bench-v1\"")
        && artifact.contains("\"command\":\"hf-cuda-decode\"")
        && artifact.contains("\"capabilities\"")
        && artifact.contains("\"summary\"")
        && artifact.contains("\"backend\":\"cuda\"")
        && artifact.contains("\"input_mode\":\"token_ids\"")
        && artifact.contains("\"resident_weight_plan\"")
        && artifact.contains("\"cuda_contract_matched\":true")
        && artifact.contains("\"host_causality_edges\":0")
        && artifact.contains("\"hot_path_allocations\":0")
        && artifact.contains("\"critical_paths\":[")
        && artifact.contains("\"proves_host_wait_not_gpu_idle\":true")
        && artifact.contains("\"token_ledgers\":[")
        && artifact.contains("\"hf_cuda_sequence_device_timeline\"")
}

fn write_tiny_hf_checkpoint_dir(prefix: &str) -> Result<PathBuf, String> {
    let dir = std::env::temp_dir().join(format!("{prefix}-{}", std::process::id()));
    std::fs::create_dir_all(&dir)
        .map_err(|err| format!("failed to create checkpoint dir {}: {err}", dir.display()))?;
    let config = tiny_hf_config();
    std::fs::write(dir.join("config.json"), config)
        .map_err(|err| format!("failed to write checkpoint config: {err}"))?;
    let metadata = nerva_model::hf::parser::parse_hf_config_metadata(config)
        .map_err(|err| format!("failed to parse checkpoint config: {err:?}"))?;
    let layout = nerva_model::weights::layout::plan::plan_hf_weight_layout(&metadata)
        .map_err(|err| format!("failed to plan checkpoint layout: {err:?}"))?;
    let manifest = nerva_model::weights::manifest::build_hf_tensor_manifest(&layout)
        .map_err(|err| format!("failed to build checkpoint manifest: {err:?}"))?;
    let header =
        nerva_model::weights::safetensors::header::synthetic_safetensors_header_for_manifest(
            &manifest,
        )
        .map_err(|err| format!("failed to build checkpoint header: {err:?}"))?;
    write_safetensors_header(
        &dir.join("model.safetensors"),
        &header,
        manifest.total_weight_bytes,
    )?;
    Ok(dir)
}

fn tiny_hf_config() -> &'static str {
    r#"{
            "model_type": "llama",
            "hidden_size": 2,
            "intermediate_size": 2,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 4,
            "torch_dtype": "float16"
        }"#
}

fn write_safetensors_header(path: &Path, header: &str, payload_bytes: usize) -> Result<(), String> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(8 + header.len() + payload_bytes, 0);
    std::fs::write(path, bytes)
        .map_err(|err| format!("failed to write safetensors file {}: {err}", path.display()))
}

fn remove_tiny_hf_checkpoint_dir(dir: &Path) {
    let _ = std::fs::remove_file(dir.join("model.safetensors"));
    let _ = std::fs::remove_file(dir.join("config.json"));
    let _ = std::fs::remove_dir(dir);
}
