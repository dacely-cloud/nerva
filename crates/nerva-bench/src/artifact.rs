use std::process::Command;

use crate::{
    acceptance::run_acceptance_probe,
    json::{json_env_string, json_escape, json_string_array},
    model_io::{
        run_hotset_probe, run_layout_probe, run_manifest_probe, run_metadata_probe,
        run_resident_shard_probe, run_resident_weight_probe, run_safetensors_probe,
        run_safetensors_shard_probe, run_weight_execution_probe,
    },
    parse::{parse_optional_u32, parse_optional_u64, parse_optional_usize},
    probes::{
        run_capabilities, run_kv_probe, run_synthetic, run_synthetic_ledger_probe,
        run_topology_probe, run_transport_matrix_probe, run_transport_probe,
    },
};

pub(crate) fn run_artifact(command: Option<String>, args: Vec<String>) -> Result<String, String> {
    let command = command.ok_or_else(|| "artifact requires a probe name".to_string())?;
    let summary = run_artifact_probe(&command, &args)?;
    Ok(format!(
        "{{\"status\":\"ok\",\"artifact_schema\":\"nerva-bench-v1\",\"metadata\":{},\"summary\":{}}}",
        artifact_metadata_json(&command, &args),
        summary
    ))
}

fn run_artifact_probe(command: &str, args: &[String]) -> Result<String, String> {
    match command {
        "smoke" => Ok(nerva_runtime::cuda_smoke().to_json()),
        "cuda-graph" => {
            let steps = parse_optional_u32(args.first().cloned(), 1024, "steps")?;
            let ring_capacity = parse_optional_u32(args.get(1).cloned(), 64, "ring_capacity")?;
            let seed_token = parse_optional_u32(args.get(2).cloned(), 1, "seed_token")?;
            Ok(
                nerva_runtime::cuda_synthetic_graph_smoke(steps, ring_capacity, seed_token)
                    .to_json(),
            )
        }
        "capabilities" => run_capabilities(),
        "topology" => run_topology_probe(),
        "synthetic" => {
            let steps = parse_optional_u64(args.first().cloned(), 1024, "steps")?;
            let ring_capacity = parse_optional_usize(args.get(1).cloned(), 64, "ring_capacity")?;
            run_synthetic(steps, ring_capacity)
        }
        "ledger" => run_synthetic_ledger_probe(),
        "block" => nerva_model::reference_block_smoke()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("reference block failed: {err:?}")),
        "model" => {
            let steps = parse_optional_usize(args.first().cloned(), 8, "steps")?;
            nerva_model::tiny_greedy_decode_smoke(steps)
                .map(|summary| summary.to_json())
                .map_err(|err| format!("tiny greedy model failed: {err:?}"))
        }
        "metadata" => run_metadata_probe(args.first().cloned()),
        "layout" => run_layout_probe(args.first().cloned()),
        "manifest" => run_manifest_probe(args.first().cloned()),
        "safetensors" => run_safetensors_probe(args.first().cloned(), args.get(1).cloned()),
        "safetensors-shards" => run_safetensors_shard_probe(
            args.first().cloned(),
            args.get(1).cloned(),
            args.get(2).cloned(),
        ),
        "resident-shards" => {
            let max_task_bytes =
                parse_optional_usize(args.get(3).cloned(), 16 * 1024 * 1024, "max_task_bytes")?;
            run_resident_shard_probe(
                args.first().cloned(),
                args.get(1).cloned(),
                args.get(2).cloned(),
                max_task_bytes,
            )
        }
        "resident-weights" => run_resident_weight_probe(args.first().cloned()),
        "hotset" => {
            let vram_bytes =
                parse_optional_usize(args.get(1).cloned(), 512 * 1024 * 1024, "vram_bytes")?;
            let max_promote_bytes =
                parse_optional_usize(args.get(2).cloned(), vram_bytes, "max_promote_bytes")?;
            run_hotset_probe(args.first().cloned(), vram_bytes, max_promote_bytes)
        }
        "weight-exec" => {
            let vram_bytes =
                parse_optional_usize(args.get(1).cloned(), 512 * 1024 * 1024, "vram_bytes")?;
            let max_promote_bytes =
                parse_optional_usize(args.get(2).cloned(), vram_bytes, "max_promote_bytes")?;
            let max_steps = parse_optional_usize(args.get(3).cloned(), 32, "max_steps")?;
            let compute_capability =
                parse_optional_u64(args.get(4).cloned(), 89, "compute_capability")?;
            run_weight_execution_probe(
                args.first().cloned(),
                vram_bytes,
                max_promote_bytes,
                max_steps,
                Some(compute_capability as u32),
            )
        }
        "attention" => nerva_model::blockwise_attention_smoke()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("blockwise attention failed: {err:?}")),
        "warm" => nerva_model::warm_compute_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("warm compute probe failed: {err:?}")),
        "contracts" => nerva_kernel_contracts::kernel_registry_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("kernel contract probe failed: {err:?}")),
        "kv" => run_kv_probe(),
        "transport" => run_transport_probe(),
        "transport-matrix" => run_transport_matrix_probe(),
        "acceptance" => run_acceptance_probe(),
        _ => Err(format!("unknown artifact probe '{command}'")),
    }
}

fn artifact_metadata_json(command: &str, args: &[String]) -> String {
    let capabilities = run_capabilities().unwrap_or_else(|reason| {
        format!(
            "{{\"status\":\"failed\",\"error\":\"{}\"}}",
            json_escape(&reason)
        )
    });
    format!(
        "{{\"command\":\"{}\",\"args\":{},\"git_commit\":\"{}\",\"package_version\":\"{}\",\"profile\":\"{}\",\"target\":\"{}-{}\",\"rustc_version\":\"{}\",\"cargo_version\":\"{}\",\"rustflags\":{},\"cargo_encoded_rustflags\":{},\"capabilities\":{}}}",
        json_escape(command),
        json_string_array(args),
        json_escape(&current_git_commit()),
        env!("CARGO_PKG_VERSION"),
        build_profile(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        json_escape(&command_version("rustc")),
        json_escape(&command_version("cargo")),
        json_env_string("RUSTFLAGS"),
        json_env_string("CARGO_ENCODED_RUSTFLAGS"),
        capabilities,
    )
}

fn current_git_commit() -> String {
    if let Some(commit) = option_env!("NERVA_GIT_COMMIT") {
        return commit.to_string();
    }
    let Ok(output) = Command::new("git").args(["rev-parse", "HEAD"]).output() else {
        return "unknown".to_string();
    };
    if !output.status.success() {
        return "unknown".to_string();
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn command_version(command: &str) -> String {
    let Ok(output) = Command::new(command).arg("--version").output() else {
        return "unknown".to_string();
    };
    if !output.status.success() {
        return "unknown".to_string();
    }
    String::from_utf8(output.stdout)
        .ok()
        .and_then(|stdout| stdout.lines().next().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}
