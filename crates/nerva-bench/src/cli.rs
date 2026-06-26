use std::process::ExitCode;

use crate::{
    acceptance::build_acceptance_report,
    artifact::run_artifact,
    model_io::{
        run_hotset_probe, run_layout_probe, run_manifest_probe, run_metadata_probe,
        run_resident_shard_probe, run_resident_weight_probe, run_safetensors_probe,
        run_safetensors_shard_probe, run_weight_execution_probe,
    },
    parity::load_vllm_token_identity_parity,
    parse::{parse_optional_u32, parse_optional_u64, parse_optional_usize},
    probes::{
        run_capabilities, run_kv_probe, run_synthetic, run_synthetic_ledger_probe,
        run_topology_probe, run_transport_matrix_probe, run_transport_probe,
    },
};

pub(crate) fn run() -> ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("smoke") => {
            let summary = nerva_runtime::capabilities::discovery::cuda_smoke();
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        Some("cuda-graph") => {
            let steps = match parse_optional_u32(args.next(), 1024, "steps") {
                Ok(steps) => steps,
                Err(reason) => {
                    eprintln!("{reason}");
                    return ExitCode::from(2);
                }
            };
            let ring_capacity = match parse_optional_u32(args.next(), 64, "ring_capacity") {
                Ok(ring_capacity) => ring_capacity,
                Err(reason) => {
                    eprintln!("{reason}");
                    return ExitCode::from(2);
                }
            };
            let seed_token = match parse_optional_u32(args.next(), 1, "seed_token") {
                Ok(seed_token) => seed_token,
                Err(reason) => {
                    eprintln!("{reason}");
                    return ExitCode::from(2);
                }
            };
            let summary = nerva_runtime::engine::cuda::cuda_synthetic_graph_smoke(
                steps,
                ring_capacity,
                seed_token,
            );
            println!("{}", summary.to_json());
            if format!("{:?}", summary.status) == "Ok" {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Some("cuda-block") => {
            let summary = nerva_runtime::engine::cuda::cuda_tiny_block_smoke();
            println!("{}", summary.to_json());
            if format!("{:?}", summary.status) == "Ok" {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Some("cuda-loaded-block") => {
            let summary = nerva_runtime::engine::cuda::cuda_loaded_tiny_block_smoke();
            println!("{}", summary.to_json());
            if format!("{:?}", summary.status) == "Ok" {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
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
            match nerva_model::tiny::tiny_greedy_decode_smoke(steps) {
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
        Some("attention") => match nerva_model::attention::blockwise_attention_smoke() {
            Ok(summary) => {
                println!("{}", summary.to_json());
                ExitCode::SUCCESS
            }
            Err(err) => {
                eprintln!("blockwise attention failed: {err:?}");
                ExitCode::from(1)
            }
        },
        Some("warm") => match nerva_model::warm_compute::warm_compute_probe() {
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
            eprintln!(
                "usage: cargo run -p nerva-bench -- smoke\n       cargo run -p nerva-bench -- cuda-graph [steps] [ring_capacity] [seed_token]\n       cargo run -p nerva-bench -- cuda-block\n       cargo run -p nerva-bench -- cuda-loaded-block\n       cargo run -p nerva-bench -- capabilities\n       cargo run -p nerva-bench -- topology\n       cargo run -p nerva-bench -- synthetic [steps] [ring_capacity]\n       cargo run -p nerva-bench -- ledger\n       cargo run -p nerva-bench -- block\n       cargo run -p nerva-bench -- precision\n       cargo run -p nerva-bench -- safetensors-block\n       cargo run -p nerva-bench -- model [steps]\n       cargo run -p nerva-bench -- vllm-parity vllm_tokens.json [steps]\n       cargo run -p nerva-bench -- metadata [config.json]\n       cargo run -p nerva-bench -- layout [config.json]\n       cargo run -p nerva-bench -- manifest [config.json]\n       cargo run -p nerva-bench -- safetensors [config.json model.safetensors]\n       cargo run -p nerva-bench -- safetensors-shards config.json model.safetensors.index.json checkpoint_dir\n       cargo run -p nerva-bench -- resident-shards config.json model.safetensors.index.json checkpoint_dir [max_task_bytes]\n       cargo run -p nerva-bench -- resident-weights [config.json]\n       cargo run -p nerva-bench -- hotset [config.json] [vram_bytes] [max_promote_bytes]\n       cargo run -p nerva-bench -- weight-exec [config.json] [vram_bytes] [max_promote_bytes] [max_steps] [compute_capability]\n       cargo run -p nerva-bench -- attention\n       cargo run -p nerva-bench -- warm\n       cargo run -p nerva-bench -- contracts\n       cargo run -p nerva-bench -- kv\n       cargo run -p nerva-bench -- transport\n       cargo run -p nerva-bench -- transport-matrix\n       cargo run -p nerva-bench -- acceptance\n       cargo run -p nerva-bench -- artifact <probe> [probe args...]"
            );
            ExitCode::from(2)
        }
    }
}
