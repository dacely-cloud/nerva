use std::process::ExitCode;

mod args;
mod model;
mod run;
mod ui;

pub(crate) fn run() -> ExitCode {
    ui::terminal::install_signal_cleanup();

    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    let json_requested = raw_args.iter().any(|arg| arg == "--json");
    let rest = match raw_args.first().map(String::as_str) {
        Some("generate" | "chat" | "ask") => &raw_args[1..],
        Some("-h" | "--help") | None => {
            print_usage();
            return ExitCode::SUCCESS;
        }
        Some(first) if first.starts_with('-') => &raw_args[..],
        Some(command) => {
            print_error(&format!("unknown command: {command}"), json_requested);
            return ExitCode::from(2);
        }
    };
    match run::run_generate(rest) {
        Ok(output) => {
            if output.print_stdout {
                println!("{}", style_stdout_output(&output.output, json_requested));
            }
            ExitCode::SUCCESS
        }
        Err(reason) => {
            print_error(&reason, json_requested);
            ExitCode::from(1)
        }
    }
}

fn print_usage() {
    for line in usage_lines() {
        eprintln!("{line}");
    }
}

fn print_error(reason: &str, _json_requested: bool) {
    eprintln!("{reason}");
    for line in usage_lines() {
        eprintln!("{line}");
    }
}

fn usage_lines() -> &'static [&'static str] {
    &[
        "usage:",
        "  cargo run -p nerva -- -m model -p prompt [-c context] [-o output] [--json] [--debug]",
        "  cargo run -p nerva -- generate -m model -p prompt [-c context] [-o output] [--debug]",
        "",
        "aliases:",
        "  generate, chat, ask",
        "",
        "notes:",
        "  default output uses plain streaming logs on stderr",
        "  --debug enables the Ratatui dashboard with charts",
        "  NERVA_COLOR=never|ansi|truecolor|always controls log color",
    ]
}

fn style_stdout_output(value: &str, _json_requested: bool) -> String {
    value.to_string()
}
