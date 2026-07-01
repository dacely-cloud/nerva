use std::process::ExitCode;

pub(crate) mod args;
pub(crate) mod model;
mod run;
mod ui;

pub(crate) fn run() -> ExitCode {
    #[cfg(unix)]
    ui::terminal::install_signal_cleanup();

    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    if raw_args.first().map(String::as_str) == Some("serve") {
        return run_serve(&raw_args[1..]);
    }
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

fn run_serve(args: &[String]) -> ExitCode {
    let parsed = match self::args::parse_serve_args(args) {
        Ok(parsed) => parsed,
        Err(reason) => {
            print_error(&reason);
            return ExitCode::from(2);
        }
    };
    let Some(model) = parsed.model.clone() else {
        print_error("missing -m/--model");
        return ExitCode::from(2);
    };
    let config = crate::openai::ServeConfig {
        model,
        host: parsed.host,
        port: parsed.port,
        context_tokens: parsed.context_tokens,
        output_tokens: parsed.output_tokens,
        queue_capacity: parsed.queue_capacity,
        compute_capability: parsed.compute_capability,
        max_concurrent_requests: parsed.max_concurrent_requests,
        workers: parsed.workers,
        max_blocking_threads: parsed.max_blocking_threads,
        api_key: parsed.api_key,
        rt: parsed.rt,
        rt_mode: parsed.rt_mode,
        rt_page_tokens: parsed.rt_page_tokens,
        rt_pages: parsed.rt_pages,
        rt_far_pages: parsed.rt_far_pages,
        rt_local_window_tokens: parsed.rt_local_window_tokens,
        rt_sink_tokens: parsed.rt_sink_tokens,
        profiling: parsed.profiling,
    };
    match crate::openai::run_server(config) {
        Ok(()) => ExitCode::SUCCESS,
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
        "  cargo run -p nerva -- serve -m model [--host 127.0.0.1] [--port 8000] [--max-concurrent-requests count] [--api-key key]",
        "",
        "aliases:",
        "  generate, chat, ask",
        "  serve starts the OpenAI-compatible HTTP API",
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
