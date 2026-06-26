use std::process::ExitCode;

use crate::cli::exit;
use crate::parity::load_vllm_token_identity_parity;
use crate::parse::parse_optional_usize;
use nerva_core::types::dtype::DType;

pub(crate) fn dispatch(
    command: Option<&str>,
    args: &mut impl Iterator<Item = String>,
) -> Option<ExitCode> {
    match command {
        Some("block") => Some(run_reference_block()),
        Some("precision") => Some(run_precision_block()),
        Some("safetensors-block") => Some(run_safetensors_block()),
        Some("model") => Some(run_tiny_model(args)),
        Some("precision-model") => Some(run_tiny_precision_model(args)),
        Some("vllm-parity") => Some(run_vllm_parity(args)),
        Some("attention") => Some(run_attention()),
        Some("warm") => Some(run_warm_compute()),
        Some("contracts") => Some(run_kernel_contracts()),
        _ => None,
    }
}

fn run_reference_block() -> ExitCode {
    match nerva_model::reference::smoke::reference_block_smoke() {
        Ok(summary) => {
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("reference block failed: {err:?}");
            ExitCode::from(1)
        }
    }
}

fn run_precision_block() -> ExitCode {
    match nerva_model::precision::smoke::precision_block_smoke() {
        Ok(summary) => {
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("precision block failed: {err:?}");
            ExitCode::from(1)
        }
    }
}

fn run_safetensors_block() -> ExitCode {
    match nerva_model::precision::file_smoke::run::precision_block_from_safetensors_smoke() {
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

fn run_tiny_model(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let steps = match parse_optional_usize(args.next(), 8, "steps") {
        Ok(steps) => steps,
        Err(reason) => return exit::parse_error(reason),
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

fn run_tiny_precision_model(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let steps = match parse_optional_usize(args.next(), 8, "steps") {
        Ok(steps) => steps,
        Err(reason) => return exit::parse_error(reason),
    };
    match precision_model_pair_json(steps) {
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

pub(crate) fn precision_model_pair_json(steps: usize) -> Result<String, String> {
    let f16 = nerva_model::tiny::precision::tiny_precision_greedy_decode_smoke(DType::F16, steps)
        .map_err(|err| format!("tiny FP16 precision model failed: {err:?}"))?;
    let bf16 = nerva_model::tiny::precision::tiny_precision_greedy_decode_smoke(DType::BF16, steps)
        .map_err(|err| format!("tiny BF16 precision model failed: {err:?}"))?;
    let passed = f16.passed() && bf16.passed() && f16.output_hash == bf16.output_hash;
    Ok(format!(
        "{{\"status\":\"{}\",\"steps\":{},\"passed\":{},\"f16\":{},\"bf16\":{}}}",
        if passed { "ok" } else { "failed" },
        steps,
        passed,
        f16.to_json(),
        bf16.to_json(),
    ))
}

fn run_vllm_parity(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let path = args.next();
    let steps = match parse_optional_usize(args.next(), 8, "steps") {
        Ok(steps) => steps,
        Err(reason) => return exit::parse_error(reason),
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

fn run_attention() -> ExitCode {
    match nerva_model::attention::smoke::blockwise_attention_smoke() {
        Ok(summary) => {
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("blockwise attention failed: {err:?}");
            ExitCode::from(1)
        }
    }
}

fn run_warm_compute() -> ExitCode {
    match nerva_model::warm_compute::probe::warm_compute_probe() {
        Ok(summary) => {
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("warm compute probe failed: {err:?}");
            ExitCode::from(1)
        }
    }
}

fn run_kernel_contracts() -> ExitCode {
    match nerva_kernel_contracts::registry::probe::kernel_registry_probe() {
        Ok(summary) => {
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("kernel contract probe failed: {err:?}");
            ExitCode::from(1)
        }
    }
}
