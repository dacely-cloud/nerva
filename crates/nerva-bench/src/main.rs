#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use std::process::ExitCode;

use nerva_core::TokenId;
use nerva_runtime::{KvResidencyProbeConfig, Runtime, RuntimeConfig, SyntheticDecodeConfig};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("smoke") => {
            let summary = nerva_runtime::cuda_smoke();
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
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
        _ => {
            eprintln!(
                "usage: cargo run -p nerva-bench -- smoke\n       cargo run -p nerva-bench -- synthetic [steps] [ring_capacity]\n       cargo run -p nerva-bench -- block\n       cargo run -p nerva-bench -- model [steps]\n       cargo run -p nerva-bench -- attention\n       cargo run -p nerva-bench -- warm\n       cargo run -p nerva-bench -- contracts\n       cargo run -p nerva-bench -- kv"
            );
            ExitCode::from(2)
        }
    }
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
