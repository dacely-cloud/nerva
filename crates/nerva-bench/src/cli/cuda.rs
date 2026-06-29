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
        Some("experimental-rt") => Some(run_experimental_rt(args)),
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
    let summary = nerva_cuda::backend::probe::backend_contract_smoke(device_bytes, pinned_bytes);
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
    let summary = nerva_cuda::graph::probe::synthetic_graph_smoke(steps, ring_capacity, seed_token);
    print_status_json(summary.to_json(), format!("{:?}", summary.status) == "Ok")
}

fn run_block() -> ExitCode {
    let summary = nerva_cuda::block::probe::tiny_block_smoke();
    print_status_json(summary.to_json(), format!("{:?}", summary.status) == "Ok")
}

fn run_loaded_block() -> ExitCode {
    let summary = nerva_cuda::block::probe::loaded_tiny_block_smoke();
    print_status_json(summary.to_json(), format!("{:?}", summary.status) == "Ok")
}

fn run_attention() -> ExitCode {
    let summary = nerva_cuda::attention::probe::tiered_attention_smoke();
    print_status_json(summary.to_json(), format!("{:?}", summary.status) == "Ok")
}

fn run_sampler() -> ExitCode {
    let summary = nerva_cuda::sampler::probe::greedy_sampler_smoke();
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
    let summary = nerva_cuda::decode::probe::tiny_decode_smoke(steps, ring_capacity, seed_token);
    print_status_json(summary.to_json(), format!("{:?}", summary.status) == "Ok")
}

fn run_experimental_rt(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let context_tokens = match parse_optional_usize(args.next(), 128 * 1024, "context_tokens") {
        Ok(context_tokens) => context_tokens,
        Err(reason) => {
            eprintln!("{reason}");
            return ExitCode::from(2);
        }
    };
    let query_count = match parse_optional_u32(args.next(), 1, "query_count") {
        Ok(query_count) => query_count,
        Err(reason) => {
            eprintln!("{reason}");
            return ExitCode::from(2);
        }
    };
    let candidates_per_query = match parse_optional_u32(args.next(), 128, "candidates_per_query") {
        Ok(candidates_per_query) => candidates_per_query,
        Err(reason) => {
            eprintln!("{reason}");
            return ExitCode::from(2);
        }
    };
    let iterations = match parse_optional_u32(args.next(), 128, "iterations") {
        Ok(iterations) => iterations,
        Err(reason) => {
            eprintln!("{reason}");
            return ExitCode::from(2);
        }
    };
    let page_tokens = 16u32;
    let dims = 16u32;
    let pages = context_tokens
        .saturating_add(page_tokens as usize - 1)
        .saturating_div(page_tokens as usize)
        .max(1)
        .min(u32::MAX as usize) as u32;
    let summary = nerva_cuda::experimental_rt::probe::experimental_rt_candidate_bench(
        pages,
        page_tokens,
        dims,
        query_count,
        candidates_per_query.min(pages).max(1),
        iterations,
        8,
    );
    let json = format!(
        "{{\"status\":\"{}\",\"backend\":\"cuda\",\"mode\":\"experimental_rt_candidate\",\"scope\":\"candidate_selector_only\",\"context_tokens\":{},\"summary\":{}}}",
        if summary.passed() { "ok" } else { "failed" },
        context_tokens,
        summary.to_json(),
    );
    print_status_json(json, summary.passed())
}

fn print_status_json(json: String, passed: bool) -> ExitCode {
    println!("{json}");
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
