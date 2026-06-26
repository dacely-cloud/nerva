use std::process::ExitCode;

use crate::cli::exit;
use crate::parse::parse_optional_usize;

pub(crate) fn run_prompt_model(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let prompt = args.next().unwrap_or_else(|| "zero".to_string());
    let steps = match parse_optional_usize(args.next(), 8, "steps") {
        Ok(steps) => steps,
        Err(reason) => return exit::parse_error(reason),
    };
    match nerva_model::prompt::decode::tiny_prompt_decode_smoke(&prompt, steps) {
        Ok(summary) => {
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("tiny prompt model failed: {err:?}");
            ExitCode::from(1)
        }
    }
}
