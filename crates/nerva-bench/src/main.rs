#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use std::{
    io::Read,
    path::Path,
    path::PathBuf,
    process::{Command, ExitCode},
};

use nerva_core::TokenId;
use nerva_runtime::{
    KvResidencyProbeConfig, ResidencyBudget, Runtime, RuntimeConfig, SyntheticDecodeConfig,
};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("smoke") => {
            let summary = nerva_runtime::cuda_smoke();
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        Some("capabilities") => match run_capabilities() {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(reason) => {
                eprintln!("{reason}");
                ExitCode::from(1)
            }
        },
        Some("synthetic") => {
            let steps = match parse_optional_u64(args.next(), 1024, "steps") {
                Ok(steps) => steps,
                Err(reason) => {
                    eprintln!("{reason}");
                    return ExitCode::from(2);
                }
            };
            let ring_capacity = match parse_optional_usize(args.next(), 64, "ring_capacity") {
                Ok(ring_capacity) => ring_capacity,
                Err(reason) => {
                    eprintln!("{reason}");
                    return ExitCode::from(2);
                }
            };
            match run_synthetic(steps, ring_capacity) {
                Ok(json) => {
                    println!("{json}");
                    ExitCode::SUCCESS
                }
                Err(reason) => {
                    eprintln!("{reason}");
                    ExitCode::from(1)
                }
            }
        }
        Some("block") => match nerva_model::reference_block_smoke() {
            Ok(summary) => {
                println!("{}", summary.to_json());
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("reference block failed: {err:?}");
                ExitCode::from(1)
            }
        },
        Some("model") => {
            let steps = match parse_optional_usize(args.next(), 8, "steps") {
                Ok(steps) => steps,
                Err(reason) => {
                    eprintln!("{reason}");
                    return ExitCode::from(2);
                }
            };
            match nerva_model::tiny_greedy_decode_smoke(steps) {
                Ok(summary) => {
                    println!("{}", summary.to_json());
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("tiny greedy model failed: {err:?}");
                    ExitCode::from(1)
                }
            }
        }
        Some("metadata") => match run_metadata_probe(args.next()) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(reason) => {
                eprintln!("{reason}");
                ExitCode::from(1)
            }
        },
        Some("layout") => match run_layout_probe(args.next()) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(reason) => {
                eprintln!("{reason}");
                ExitCode::from(1)
            }
        },
        Some("manifest") => match run_manifest_probe(args.next()) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(reason) => {
                eprintln!("{reason}");
                ExitCode::from(1)
            }
        },
        Some("safetensors") => match run_safetensors_probe(args.next(), args.next()) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(reason) => {
                eprintln!("{reason}");
                ExitCode::from(1)
            }
        },
        Some("safetensors-shards") => {
            match run_safetensors_shard_probe(args.next(), args.next(), args.next()) {
                Ok(json) => {
                    println!("{json}");
                    ExitCode::SUCCESS
                }
                Err(reason) => {
                    eprintln!("{reason}");
                    ExitCode::from(1)
                }
            }
        }
        Some("resident-shards") => {
            let config_path = args.next();
            let index_path = args.next();
            let checkpoint_dir = args.next();
            let max_task_bytes =
                match parse_optional_usize(args.next(), 16 * 1024 * 1024, "max_task_bytes") {
                    Ok(value) => value,
                    Err(reason) => {
                        eprintln!("{reason}");
                        return ExitCode::from(2);
                    }
                };
            match run_resident_shard_probe(config_path, index_path, checkpoint_dir, max_task_bytes)
            {
                Ok(json) => {
                    println!("{json}");
                    ExitCode::SUCCESS
                }
                Err(reason) => {
                    eprintln!("{reason}");
                    ExitCode::from(1)
                }
            }
        }
        Some("resident-weights") => match run_resident_weight_probe(args.next()) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(reason) => {
                eprintln!("{reason}");
                ExitCode::from(1)
            }
        },
        Some("hotset") => {
            let config_path = args.next();
            let vram_bytes =
                match parse_optional_usize(args.next(), 512 * 1024 * 1024, "vram_bytes") {
                    Ok(value) => value,
                    Err(reason) => {
                        eprintln!("{reason}");
                        return ExitCode::from(2);
                    }
                };
            let max_promote_bytes =
                match parse_optional_usize(args.next(), vram_bytes, "max_promote_bytes") {
                    Ok(value) => value,
                    Err(reason) => {
                        eprintln!("{reason}");
                        return ExitCode::from(2);
                    }
                };
            match run_hotset_probe(config_path, vram_bytes, max_promote_bytes) {
                Ok(json) => {
                    println!("{json}");
                    ExitCode::SUCCESS
                }
                Err(reason) => {
                    eprintln!("{reason}");
                    ExitCode::from(1)
                }
            }
        }
        Some("weight-exec") => {
            let config_path = args.next();
            let vram_bytes =
                match parse_optional_usize(args.next(), 512 * 1024 * 1024, "vram_bytes") {
                    Ok(value) => value,
                    Err(reason) => {
                        eprintln!("{reason}");
                        return ExitCode::from(2);
                    }
                };
            let max_promote_bytes =
                match parse_optional_usize(args.next(), vram_bytes, "max_promote_bytes") {
                    Ok(value) => value,
                    Err(reason) => {
                        eprintln!("{reason}");
                        return ExitCode::from(2);
                    }
                };
            let max_steps = match parse_optional_usize(args.next(), 32, "max_steps") {
                Ok(value) => value,
                Err(reason) => {
                    eprintln!("{reason}");
                    return ExitCode::from(2);
                }
            };
            let compute_capability = match parse_optional_u64(args.next(), 89, "compute_capability")
            {
                Ok(value) => value,
                Err(reason) => {
                    eprintln!("{reason}");
                    return ExitCode::from(2);
                }
            };
            match run_weight_execution_probe(
                config_path,
                vram_bytes,
                max_promote_bytes,
                max_steps,
                Some(compute_capability as u32),
            ) {
                Ok(json) => {
                    println!("{json}");
                    ExitCode::SUCCESS
                }
                Err(reason) => {
                    eprintln!("{reason}");
                    ExitCode::from(1)
                }
            }
        }
        Some("attention") => match nerva_model::blockwise_attention_smoke() {
            Ok(summary) => {
                println!("{}", summary.to_json());
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("blockwise attention failed: {err:?}");
                ExitCode::from(1)
            }
        },
        Some("warm") => match nerva_model::warm_compute_probe() {
            Ok(summary) => {
                println!("{}", summary.to_json());
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("warm compute probe failed: {err:?}");
                ExitCode::from(1)
            }
        },
        Some("contracts") => match nerva_kernel_contracts::kernel_registry_probe() {
            Ok(summary) => {
                println!("{}", summary.to_json());
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("kernel contract probe failed: {err:?}");
                ExitCode::from(1)
            }
        },
        Some("kv") => match run_kv_probe() {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(reason) => {
                eprintln!("{reason}");
                ExitCode::from(1)
            }
        },
        Some("transport") => match run_transport_probe() {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(reason) => {
                eprintln!("{reason}");
                ExitCode::from(1)
            }
        },
        Some("artifact") => match run_artifact(args.next(), args.collect()) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(reason) => {
                eprintln!("{reason}");
                ExitCode::from(1)
            }
        },
        _ => {
            eprintln!(
                "usage: cargo run -p nerva-bench -- smoke\n       cargo run -p nerva-bench -- capabilities\n       cargo run -p nerva-bench -- synthetic [steps] [ring_capacity]\n       cargo run -p nerva-bench -- block\n       cargo run -p nerva-bench -- model [steps]\n       cargo run -p nerva-bench -- metadata [config.json]\n       cargo run -p nerva-bench -- layout [config.json]\n       cargo run -p nerva-bench -- manifest [config.json]\n       cargo run -p nerva-bench -- safetensors [config.json model.safetensors]\n       cargo run -p nerva-bench -- safetensors-shards config.json model.safetensors.index.json checkpoint_dir\n       cargo run -p nerva-bench -- resident-shards config.json model.safetensors.index.json checkpoint_dir [max_task_bytes]\n       cargo run -p nerva-bench -- resident-weights [config.json]\n       cargo run -p nerva-bench -- hotset [config.json] [vram_bytes] [max_promote_bytes]\n       cargo run -p nerva-bench -- weight-exec [config.json] [vram_bytes] [max_promote_bytes] [max_steps] [compute_capability]\n       cargo run -p nerva-bench -- attention\n       cargo run -p nerva-bench -- warm\n       cargo run -p nerva-bench -- contracts\n       cargo run -p nerva-bench -- kv\n       cargo run -p nerva-bench -- transport\n       cargo run -p nerva-bench -- artifact <probe> [probe args...]"
            );
            ExitCode::from(2)
        }
    }
}

fn run_capabilities() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    Ok(runtime.discover_capabilities().to_json())
}

fn run_synthetic(steps: u64, ring_capacity: usize) -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_synthetic_decode(SyntheticDecodeConfig::new(steps, ring_capacity, TokenId(1)))
        .map_err(|err| format!("synthetic decode failed: {err:?}"))?;
    Ok(summary.to_json())
}

fn run_kv_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_kv_residency_probe(KvResidencyProbeConfig::default())
        .map_err(|err| format!("KV residency probe failed: {err:?}"))?;
    Ok(summary.to_json())
}

fn run_transport_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_transport_path_probe()
        .map_err(|err| format!("transport path probe failed: {err:?}"))?;
    Ok(summary.to_json())
}

fn run_artifact(command: Option<String>, args: Vec<String>) -> Result<String, String> {
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
        "capabilities" => run_capabilities(),
        "synthetic" => {
            let steps = parse_optional_u64(args.first().cloned(), 1024, "steps")?;
            let ring_capacity = parse_optional_usize(args.get(1).cloned(), 64, "ring_capacity")?;
            run_synthetic(steps, ring_capacity)
        }
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
        "{{\"command\":\"{}\",\"args\":{},\"git_commit\":\"{}\",\"package_version\":\"{}\",\"profile\":\"{}\",\"target\":\"{}-{}\",\"capabilities\":{}}}",
        json_escape(command),
        json_string_array(args),
        json_escape(&current_git_commit()),
        env!("CARGO_PKG_VERSION"),
        build_profile(),
        std::env::consts::OS,
        std::env::consts::ARCH,
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

fn json_string_array(values: &[String]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&json_escape(value));
        out.push('"');
    }
    out.push(']');
    out
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => escaped.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped
}

fn run_metadata_probe(config_path: Option<String>) -> Result<String, String> {
    match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            let metadata = nerva_model::parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            Ok(metadata.to_json())
        }
        None => nerva_model::hf_metadata_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("HF metadata probe failed: {err:?}")),
    }
}

fn run_layout_probe(config_path: Option<String>) -> Result<String, String> {
    match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            let metadata = nerva_model::parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            let plan = nerva_model::plan_hf_weight_layout(&metadata)
                .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
            Ok(plan.to_json())
        }
        None => nerva_model::hf_weight_layout_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("HF weight layout probe failed: {err:?}")),
    }
}

fn run_manifest_probe(config_path: Option<String>) -> Result<String, String> {
    match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            let metadata = nerva_model::parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            let plan = nerva_model::plan_hf_weight_layout(&metadata)
                .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
            let manifest = nerva_model::build_hf_tensor_manifest(&plan)
                .map_err(|err| format!("HF tensor manifest failed: {err:?}"))?;
            Ok(manifest.to_json())
        }
        None => nerva_model::hf_tensor_manifest_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("HF tensor manifest probe failed: {err:?}")),
    }
}

fn run_safetensors_probe(
    config_path: Option<String>,
    safetensors_path: Option<String>,
) -> Result<String, String> {
    match (config_path, safetensors_path) {
        (None, None) => nerva_model::safetensors_header_probe()
            .map(|summary| summary.to_json())
            .map_err(|err| format!("safetensors header probe failed: {err:?}")),
        (Some(config_path), Some(safetensors_path)) => {
            let config = std::fs::read_to_string(&config_path)
                .map_err(|err| format!("failed to read {config_path}: {err}"))?;
            let metadata = nerva_model::parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            let plan = nerva_model::plan_hf_weight_layout(&metadata)
                .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
            let manifest = nerva_model::build_hf_tensor_manifest(&plan)
                .map_err(|err| format!("HF tensor manifest failed: {err:?}"))?;
            let bytes = std::fs::read(&safetensors_path)
                .map_err(|err| format!("failed to read {safetensors_path}: {err}"))?;
            let header = nerva_model::safetensors_header_from_bytes(&bytes)
                .map_err(|err| format!("safetensors header read failed: {err:?}"))?;
            let validation =
                nerva_model::validate_safetensors_header_for_manifest(header, &manifest)
                    .map_err(|err| format!("safetensors manifest validation failed: {err:?}"))?;
            Ok(validation.to_json())
        }
        _ => Err(
            "safetensors requires either no args or both config.json and model.safetensors"
                .to_string(),
        ),
    }
}

fn run_safetensors_shard_probe(
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

fn run_resident_shard_probe(
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
        table.registry.used_bytes(nerva_core::MemoryTier::Dram),
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
) -> Result<nerva_model::SafetensorsShardPlan, String> {
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
    let metadata = nerva_model::parse_hf_config_metadata(&config)
        .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
    let plan = nerva_model::plan_hf_weight_layout(&metadata)
        .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
    let manifest = nerva_model::build_hf_tensor_manifest(&plan)
        .map_err(|err| format!("HF tensor manifest failed: {err:?}"))?;
    let index_json = std::fs::read_to_string(&index_path)
        .map_err(|err| format!("failed to read {index_path}: {err}"))?;
    let shard_files = nerva_model::required_safetensors_shards_for_manifest(&index_json, &manifest)
        .map_err(|err| format!("safetensors index validation failed: {err:?}"))?;
    let checkpoint_dir = PathBuf::from(checkpoint_dir);
    let mut shard_headers = Vec::with_capacity(shard_files.len());
    for shard_file in shard_files {
        let header = read_safetensors_header_only(&checkpoint_dir.join(&shard_file))?;
        shard_headers.push((shard_file, header));
    }
    let shard_header_refs = shard_headers
        .iter()
        .map(|(file_name, header_json)| {
            nerva_model::SafetensorsShardHeader::new(file_name, header_json)
        })
        .collect::<Vec<_>>();
    nerva_model::plan_safetensors_shards_for_manifest(&index_json, &shard_header_refs, &manifest)
        .map_err(|err| format!("safetensors shard plan failed: {err:?}"))
}

fn read_safetensors_header_only(path: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes).map_err(|err| {
        format!(
            "failed to read safetensors header length from {}: {err}",
            path.display()
        )
    })?;
    let header_len = usize::try_from(u64::from_le_bytes(header_len_bytes)).map_err(|_| {
        format!(
            "safetensors header length in {} does not fit usize",
            path.display()
        )
    })?;
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header).map_err(|err| {
        format!(
            "failed to read safetensors header from {}: {err}",
            path.display()
        )
    })?;
    String::from_utf8(header).map_err(|err| {
        format!(
            "safetensors header in {} is not UTF-8: {err}",
            path.display()
        )
    })
}

fn run_resident_weight_probe(config_path: Option<String>) -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            let metadata = nerva_model::parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            let plan = nerva_model::plan_hf_weight_layout(&metadata)
                .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
            let manifest = nerva_model::build_hf_tensor_manifest(&plan)
                .map_err(|err| format!("HF tensor manifest failed: {err:?}"))?;
            let table = runtime
                .materialize_hf_weight_manifest(&manifest)
                .map_err(|err| format!("resident weight materialization failed: {err:?}"))?;
            Ok(format!(
                "{{\"status\":\"ok\",\"blocks\":{},\"total_weight_bytes\":{},\"dram_used_bytes\":{},\"manifest_hash\":{},\"hot_path_allocations\":{}}}",
                table.entries.len(),
                table.total_weight_bytes,
                table.registry.used_bytes(nerva_core::MemoryTier::Dram),
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

fn run_hotset_probe(
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

fn run_weight_execution_probe(
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
) -> Result<nerva_model::HfTensorManifest, String> {
    match config_path {
        Some(path) => {
            let config = std::fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {path}: {err}"))?;
            let metadata = nerva_model::parse_hf_config_metadata(&config)
                .map_err(|err| format!("HF metadata parse failed: {err:?}"))?;
            let plan = nerva_model::plan_hf_weight_layout(&metadata)
                .map_err(|err| format!("HF weight layout failed: {err:?}"))?;
            nerva_model::build_hf_tensor_manifest(&plan)
                .map_err(|err| format!("HF tensor manifest failed: {err:?}"))
        }
        None => nerva_model::hf_tensor_manifest_probe()
            .map(|summary| summary.manifest)
            .map_err(|err| format!("HF tensor manifest probe failed: {err:?}")),
    }
}

fn parse_optional_u64(
    value: Option<String>,
    default: u64,
    label: &'static str,
) -> Result<u64, String> {
    match value {
        Some(value) => value
            .parse::<u64>()
            .map_err(|_| format!("{label} must be an unsigned integer")),
        None => Ok(default),
    }
}

fn parse_optional_usize(
    value: Option<String>,
    default: usize,
    label: &'static str,
) -> Result<usize, String> {
    match value {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{label} must be an unsigned integer")),
        None => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHARD_ONE: &str = "model-00001-of-00002.safetensors";
    const SHARD_TWO: &str = "model-00002-of-00002.safetensors";

    #[test]
    fn reads_only_safetensors_header_from_file() {
        let dir =
            std::env::temp_dir().join(format!("nerva-bench-header-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.safetensors");
        let header = "{\"x\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0xaa, 0xbb]);
        std::fs::write(&path, bytes).unwrap();

        assert_eq!(read_safetensors_header_only(&path).unwrap(), header);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn safetensors_shard_probe_reads_index_and_headers() {
        let dir =
            std::env::temp_dir().join(format!("nerva-bench-shard-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.json");
        let index_path = dir.join("model.safetensors.index.json");
        let config = r#"{
            "model_type": "llama",
            "hidden_size": 4,
            "intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 10,
            "torch_dtype": "float16"
        }"#;
        std::fs::write(&config_path, config).unwrap();

        let metadata = nerva_model::parse_hf_config_metadata(config).unwrap();
        let layout = nerva_model::plan_hf_weight_layout(&metadata).unwrap();
        let manifest = nerva_model::build_hf_tensor_manifest(&layout).unwrap();
        let index = synthetic_index_json(&manifest, 10);
        std::fs::write(&index_path, index).unwrap();
        write_safetensors_header(
            &dir.join(SHARD_ONE),
            &synthetic_header_for_entries(manifest.architecture, &manifest.entries[..10]),
        );
        write_safetensors_header(
            &dir.join(SHARD_TWO),
            &synthetic_header_for_entries(manifest.architecture, &manifest.entries[10..]),
        );

        let json = run_safetensors_shard_probe(
            Some(config_path.to_string_lossy().into_owned()),
            Some(index_path.to_string_lossy().into_owned()),
            Some(dir.to_string_lossy().into_owned()),
        )
        .unwrap();

        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"entries\":20"));
        assert!(json.contains("\"shards\":2"));

        let resident_json = run_resident_shard_probe(
            Some(config_path.to_string_lossy().into_owned()),
            Some(index_path.to_string_lossy().into_owned()),
            Some(dir.to_string_lossy().into_owned()),
            128,
        )
        .unwrap();
        assert!(resident_json.contains("\"status\":\"ok\""));
        assert!(resident_json.contains("\"blocks\":20"));
        assert!(resident_json.contains("\"prefetch\""));
        assert!(resident_json.contains("\"execution\""));
        assert!(resident_json.contains("\"tasks\":20"));

        let hotset_json =
            run_hotset_probe(Some(config_path.to_string_lossy().into_owned()), 256, 200).unwrap();
        assert!(hotset_json.contains("\"status\":\"ok\""));
        assert!(hotset_json.contains("\"hotset\""));
        assert!(hotset_json.contains("\"promoted_blocks\":7"));

        let execution_json = run_weight_execution_probe(
            Some(config_path.to_string_lossy().into_owned()),
            128,
            100,
            3,
            Some(89),
        )
        .unwrap();
        assert!(execution_json.contains("\"status\":\"ok\""));
        assert!(execution_json.contains("\"execution\""));
        assert!(execution_json.contains("\"run\""));
        assert!(execution_json.contains("\"gpu_resident_steps\":2"));
        assert!(execution_json.contains("\"gpu_staged_steps\":1"));

        let _ = std::fs::remove_file(dir.join(SHARD_ONE));
        let _ = std::fs::remove_file(dir.join(SHARD_TWO));
        let _ = std::fs::remove_file(config_path);
        let _ = std::fs::remove_file(index_path);
        let _ = std::fs::remove_dir(dir);
    }

    #[test]
    fn artifact_wraps_probe_with_reproducibility_metadata() {
        let json = run_artifact(
            Some("synthetic".to_string()),
            vec!["2".to_string(), "4".to_string()],
        )
        .unwrap();

        assert!(json.contains("\"artifact_schema\":\"nerva-bench-v1\""));
        assert!(json.contains("\"command\":\"synthetic\""));
        assert!(json.contains("\"args\":[\"2\",\"4\"]"));
        assert!(json.contains("\"git_commit\""));
        assert!(json.contains("\"package_version\""));
        assert!(json.contains("\"capabilities\""));
        assert!(json.contains("\"target_os\":\"linux\""));
        assert!(json.contains("\"summary\""));
        assert!(json.contains("\"device_timeline_idle_ns\":0"));
    }

    #[test]
    fn json_string_array_escapes_probe_args() {
        let args = vec!["quote\"".to_string(), "line\nbreak".to_string()];
        assert_eq!(json_string_array(&args), "[\"quote\\\"\",\"line\\nbreak\"]");
    }

    fn synthetic_header_for_entries(
        architecture: nerva_model::HfArchitectureKind,
        entries: &[nerva_model::HfTensorManifestEntry],
    ) -> String {
        let total_weight_bytes = entries.iter().map(|entry| entry.bytes).sum();
        let manifest = nerva_model::HfTensorManifest {
            architecture,
            entries: entries.to_vec(),
            total_weight_bytes,
            manifest_hash: 0,
        };
        nerva_model::synthetic_safetensors_header_for_manifest(&manifest).unwrap()
    }

    fn synthetic_index_json(manifest: &nerva_model::HfTensorManifest, split_at: usize) -> String {
        let mut out = format!(
            "{{\"metadata\":{{\"total_size\":{}}},\"weight_map\":{{",
            manifest.total_weight_bytes
        );
        for (index, entry) in manifest.entries.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            out.push('"');
            out.push_str(&entry.name);
            out.push_str("\":\"");
            out.push_str(if index < split_at {
                SHARD_ONE
            } else {
                SHARD_TWO
            });
            out.push('"');
        }
        out.push_str("}}");
        out
    }

    fn write_safetensors_header(path: &Path, header: &str) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        std::fs::write(path, bytes).unwrap();
    }
}
