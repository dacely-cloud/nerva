use std::process::ExitCode;

use crate::cli::exit;
use crate::parity::run::{load_token_identity_artifact_parity, load_vllm_token_identity_parity};
use crate::parse::parse_optional_usize;

pub(crate) fn run_vllm_parity(args: &mut impl Iterator<Item = String>) -> ExitCode {
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

pub(crate) fn run_token_parity(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let baseline_path = args.next();
    let candidate_path = args.next();
    match load_token_identity_artifact_parity(baseline_path, candidate_path) {
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
