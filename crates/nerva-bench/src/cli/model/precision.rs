use std::process::ExitCode;

use crate::cli::exit;
use crate::parse::parse_optional_usize;
use nerva_core::types::dtype::DType;

pub(crate) fn run_precision_block() -> ExitCode {
    match nerva_model::precision::smoke::run::precision_block_smoke() {
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

pub(crate) fn run_tiny_precision_model(args: &mut impl Iterator<Item = String>) -> ExitCode {
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
    let f16 =
        nerva_model::tiny::precision::smoke::tiny_precision_greedy_decode_smoke(DType::F16, steps)
            .map_err(|err| format!("tiny FP16 precision model failed: {err:?}"))?;
    let bf16 =
        nerva_model::tiny::precision::smoke::tiny_precision_greedy_decode_smoke(DType::BF16, steps)
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
