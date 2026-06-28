use std::process::ExitCode;

use crate::acceptance::runner::build_acceptance_report;
use crate::artifact::run::run_artifact;
use crate::cli::{cuda, exit, model, probes, usage, weights_io};

pub(crate) fn run() -> ExitCode {
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    if is_runtime_command(&raw_args) {
        eprintln!(
            "nerva-bench is for benchmarks and artifacts. Use: cargo run -p nerva -- -m model -p prompt [-c context] [-o output]"
        );
        return ExitCode::from(2);
    }
    let mut args = raw_args.iter().skip(1).cloned();
    let command = raw_args.first().cloned();

    if let Some(exit_code) = cuda::dispatch(command.as_deref(), &mut args) {
        return exit_code;
    }
    if let Some(exit_code) = probes::dispatch(command.as_deref(), &mut args) {
        return exit_code;
    }
    if let Some(exit_code) = model::dispatch::dispatch(command.as_deref(), &mut args) {
        return exit_code;
    }
    if let Some(exit_code) = weights_io::dispatch(command.as_deref(), &mut args) {
        return exit_code;
    }

    match command.as_deref() {
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
        Some("artifact") => exit::print_json_result(run_artifact(args.next(), args.collect())),
        _ => {
            usage::print_usage();
            ExitCode::from(2)
        }
    }
}

fn is_runtime_command(args: &[String]) -> bool {
    matches!(
        args.first().map(String::as_str),
        Some("-m" | "--model" | "generate" | "chat" | "ask")
    )
}
