use std::process::ExitCode;

use crate::cli::exit;
use crate::parity::run::load_vllm_token_identity_parity;
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
