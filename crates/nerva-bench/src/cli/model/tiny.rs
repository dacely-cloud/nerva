use std::process::ExitCode;

use crate::cli::exit;
use crate::parse::parse_optional_usize;

pub(crate) fn run_tiny_model(args: &mut impl Iterator<Item = String>) -> ExitCode {
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
