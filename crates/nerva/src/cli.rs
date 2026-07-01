use std::process::ExitCode;

mod args;
mod model;
mod run;
mod ui;

pub(crate) fn run() -> ExitCode {
    #[cfg(unix)]
    ui::terminal::install_signal_cleanup();

    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    let rest = match raw_args.first().map(String::as_str) {
        Some("generate" | "chat" | "ask") => &raw_args[1..],
        Some("-h" | "--help") | None => {
            print_usage();
            return ExitCode::SUCCESS;
        }
        Some(first) if first.starts_with('-') => &raw_args[..],
        Some(command) => {
            print_error(&format!("unknown command: {command}"));
            return ExitCode::from(2);
        }
    };
    match run::run_generate(rest) {
        Ok(output) => {
            if output.print_stdout {
                println!("{}", output.output);
            }
            ExitCode::SUCCESS
        }
        Err(reason) => {
            print_error(&reason);
            ExitCode::from(1)
        }
    }
}

fn print_usage() {
    for line in usage_lines() {
        eprintln!("{line}");
    }
}

fn print_error(reason: &str) {
    eprintln!("{reason}");
    for line in usage_lines() {
        eprintln!("{line}");
    }
}

fn usage_lines() -> &'static [&'static str] {
    &[
        "usage:",
        "  cargo run -p nerva -- -m model -p prompt [-c context] [-o output] [--temperature value] [--top-p value] [--top-k value] [--seed value] [--json] [--debug] [--profiling]",
        "  cargo run -p nerva -- generate -m model -p prompt [-c context] [-o output] [--temperature value] [--top-p value] [--top-k value] [--seed value] [--debug] [--profiling]",
        "",
        "aliases:",
        "  generate, chat, ask",
        "",
        "notes:",
        "  default output uses plain streaming logs on stderr",
        "  --debug enables the Ratatui dashboard with charts",
        "  --profiling enables detailed CUDA timing buckets",
        "  default sampling is accuracy-first greedy decode",
        "  --temperature >0 enables stochastic sampling",
        "  NERVA_COLOR=never|ansi|truecolor|always controls log color",
    ]
}
