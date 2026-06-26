use std::process::ExitCode;

mod cuda;
mod usage;

use crate::{
    acceptance::build_acceptance_report,
    artifact::run_artifact,
    model_io::{
        run_hotset_probe, run_layout_probe, run_manifest_probe, run_metadata_probe,
        run_resident_shard_probe, run_resident_weight_probe, run_safetensors_probe,
        run_safetensors_shard_probe, run_weight_execution_probe,
    },
    parity::load_vllm_token_identity_parity,
    parse::{parse_optional_u64, parse_optional_usize},
    probes::{
        run_capabilities, run_kv_probe, run_synthetic, run_synthetic_ledger_probe,
        run_topology_probe, run_transport_matrix_probe, run_transport_probe,
    },
};

pub(crate) fn run() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let command = args.next();
    if let Some(exit_code) = cuda::dispatch(command.as_deref(), &mut args) {
        return exit_code;
    }

    match command.as_deref() {
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
        Some("topology") => match run_topology_probe() {
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
        Some("ledger") => match run_synthetic_ledger_probe() {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(reason) => {
                eprintln!("{reason}");
                ExitCode::from(1)
            }
        },
        Some("block") => match nerva_model::reference::smoke::reference_block_smoke() {
            Ok(summary) => {
                println!("{}", summary.to_json());
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("reference block failed: {err:?}");
                ExitCode::from(1)
            }
        },
        Some("precision") => match nerva_model::precision::smoke::precision_block_smoke() {
            Ok(summary) => {
                println!("{}", summary.to_json());
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("precision block failed: {err:?}");
                ExitCode::from(1)
            }
        },
        Some("safetensors-block") => {
            match nerva_model::precision::file_smoke::precision_block_from_safetensors_smoke() {
                Ok(summary) => {
                    println!("{}", summary.to_json());
                    ExitCode::SUCCESS
                }
                Err(err) => {
                    eprintln!("safetensors precision block failed: {err:?}");
                    ExitCode::from(1)
                }
            }
        }
        Some("model") => {
            let steps = match parse_optional_usize(args.next(), 8, "steps") {
                Ok(steps) => steps,
                Err(reason) => {
                    eprintln!("{reason}");
                    return ExitCode::from(2);
                }
            };
            match nerva_model::tiny::smoke::tiny_greedy_decode_smoke(steps) {
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
        Some("vllm-parity") => {
            let path = args.next();
            let steps = match parse_optional_usize(args.next(), 8, "steps") {
                Ok(steps) => steps,
                Err(reason) => {
                    eprintln!("{reason}");
                    return ExitCode::from(2);
                }
            };
            match load_vllm_token_identity_parity(path, steps) {
                Ok(summary) => {
                    let passed = summary.passed();
                    println!("{}", summary.to_json());
                    if passed {
                        ExitCode::SUCCESS
                    } else {
                        ExitCode::from(1)
                    }
                }
                Err(reason) => {
                    eprintln!("{reason}");
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
        Some("attention") => match nerva_model::attention::smoke::blockwise_attention_smoke() {
            Ok(summary) => {
                println!("{}", summary.to_json());
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("blockwise attention failed: {err:?}");
                ExitCode::from(1)
            }
        },
        Some("warm") => match nerva_model::warm_compute::probe::warm_compute_probe() {
            Ok(summary) => {
                println!("{}", summary.to_json());
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("warm compute probe failed: {err:?}");
                ExitCode::from(1)
            }
        },
        Some("contracts") => match nerva_kernel_contracts::registry::kernel_registry_probe() {
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
        Some("transport-matrix") => match run_transport_matrix_probe() {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(reason) => {
                eprintln!("{reason}");
                ExitCode::from(1)
            }
        },
        Some("acceptance") => match build_acceptance_report() {
            Ok(report) => {
                let passed = report.passed();
                println!("{}", report.to_json());
                if passed {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from(1)
                }
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
            usage::print_usage();
            ExitCode::from(2)
        }
    }
}
