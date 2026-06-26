use std::process::ExitCode;

use crate::parse::{parse_optional_u32, parse_optional_usize};

pub(crate) fn dispatch(
    command: Option<&str>,
    args: &mut impl Iterator<Item = String>,
) -> Option<ExitCode> {
    match command {
        Some("smoke") => Some(run_smoke()),
        Some("cuda-backend") => Some(run_backend(args)),
        Some("cuda-graph") => Some(run_graph(args)),
        Some("cuda-block") => Some(run_block()),
        Some("cuda-loaded-block") => Some(run_loaded_block()),
        Some("cuda-attention") => Some(run_attention()),
        Some("cuda-sampler") => Some(run_sampler()),
        Some("cuda-tiny-decode") => Some(run_tiny_decode(args)),
        _ => None,
    }
}

fn run_backend(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let device_bytes = match parse_optional_usize(args.next(), 4096, "device_bytes") {
        Ok(device_bytes) => device_bytes,
        Err(reason) => {
            eprintln!("{reason}");
            return ExitCode::from(2);
        }
    };
    let pinned_bytes = match parse_optional_usize(args.next(), 4096, "pinned_bytes") {
        Ok(pinned_bytes) => pinned_bytes,
        Err(reason) => {
            eprintln!("{reason}");
            return ExitCode::from(2);
        }
    };
    let summary =
        nerva_runtime::engine::cuda::cuda_backend_contract_smoke(device_bytes, pinned_bytes);
    print_status_json(summary.to_json(), summary.passed())
}

fn run_smoke() -> ExitCode {
    let summary = nerva_runtime::capabilities::discovery::cuda_smoke();
    println!("{}", summary.to_json());
    ExitCode::SUCCESS
}

fn run_graph(args: &mut impl Iterator<Item = String>) -> ExitCode {
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
    let summary =
        nerva_runtime::engine::cuda::cuda_synthetic_graph_smoke(steps, ring_capacity, seed_token);
    print_status_json(summary.to_json(), format!("{:?}", summary.status) == "Ok")
}

fn run_block() -> ExitCode {
    let summary = nerva_runtime::engine::cuda::cuda_tiny_block_smoke();
    print_status_json(summary.to_json(), format!("{:?}", summary.status) == "Ok")
}

fn run_loaded_block() -> ExitCode {
    let summary = nerva_runtime::engine::cuda::cuda_loaded_tiny_block_smoke();
    print_status_json(summary.to_json(), format!("{:?}", summary.status) == "Ok")
}

fn run_attention() -> ExitCode {
    let summary = nerva_runtime::engine::cuda::cuda_tiered_attention_smoke();
    print_status_json(summary.to_json(), format!("{:?}", summary.status) == "Ok")
}

fn run_sampler() -> ExitCode {
    let summary = nerva_runtime::engine::cuda::cuda_greedy_sampler_smoke();
    print_status_json(summary.to_json(), format!("{:?}", summary.status) == "Ok")
}

fn run_tiny_decode(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let steps = match parse_optional_u32(args.next(), 8, "steps") {
        Ok(steps) => steps,
        Err(reason) => {
            eprintln!("{reason}");
            return ExitCode::from(2);
        }
    };
    let ring_capacity = match parse_optional_u32(args.next(), 4, "ring_capacity") {
        Ok(ring_capacity) => ring_capacity,
        Err(reason) => {
            eprintln!("{reason}");
            return ExitCode::from(2);
        }
    };
    let seed_token = match parse_optional_u32(args.next(), 0, "seed_token") {
        Ok(seed_token) => seed_token,
        Err(reason) => {
            eprintln!("{reason}");
            return ExitCode::from(2);
        }
    };
    let summary =
        nerva_runtime::engine::cuda::cuda_tiny_decode_smoke(steps, ring_capacity, seed_token);
    print_status_json(summary.to_json(), format!("{:?}", summary.status) == "Ok")
}

fn print_status_json(json: String, passed: bool) -> ExitCode {
    println!("{json}");
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
