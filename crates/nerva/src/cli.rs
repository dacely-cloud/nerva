use std::process::ExitCode;

pub(crate) mod args;
pub(crate) mod model;
mod run;
mod ui;

pub(crate) fn run() -> ExitCode {
    #[cfg(unix)]
    ui::terminal::install_signal_cleanup();

    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    if raw_args
        .first()
        .map(String::as_str)
        .is_some_and(is_serve_command)
    {
        return run_serve(&raw_args[1..]);
    }
    let rest = match raw_args.first().map(String::as_str) {
        Some("run" | "generate" | "chat" | "ask") => &raw_args[1..],
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
    if is_help_request(args) {
        print_serve_usage();
        return ExitCode::SUCCESS;
    }
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

fn is_serve_command(command: &str) -> bool {
    matches!(command, "serve" | "server")
}

fn is_help_request(args: &[String]) -> bool {
    args.iter()
        .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
}

fn print_usage() {
    for line in usage_lines() {
        eprintln!("{line}");
    }
}

fn print_serve_usage() {
    for line in serve_usage_lines() {
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
        "  nerva -m model -p prompt [-c context] [-o output] [--temperature value] [--top-p value] [--top-k value] [--seed value] [--json] [--debug] [--profiling]",
        "  nerva run -m model -p prompt [-c context] [-o output] [--temperature value] [--top-p value] [--top-k value] [--seed value] [--thinking] [--profiling]",
        "  nerva generate -m model -p prompt [-c context] [-o output] [--temperature value] [--top-p value] [--top-k value] [--seed value] [--debug] [--profiling]",
        "  nerva serve -m model [--host 127.0.0.1] [--port 8000] [--max-concurrent-requests count] [--api-key key]",
        "",
        "cargo dev equivalents:",
        "  cargo run -p nerva -- -m model -p prompt [-c context] [-o output] [--temperature value] [--top-p value] [--top-k value] [--seed value] [--json] [--debug] [--profiling]",
        "  cargo run -p nerva -- run -m model -p prompt [-c context] [-o output] [--temperature value] [--top-p value] [--top-k value] [--seed value] [--thinking] [--profiling]",
        "  cargo run -p nerva -- generate -m model -p prompt [-c context] [-o output] [--temperature value] [--top-p value] [--top-k value] [--seed value] [--debug] [--profiling]",
        "  cargo run -p nerva -- serve -m model [--host 127.0.0.1] [--port 8000] [--max-concurrent-requests count] [--api-key key]",
        "",
        "aliases:",
        "  run, generate, chat, ask",
        "  serve, server start the OpenAI-compatible HTTP API",
        "",
        "notes:",
        "  default output uses plain streaming logs on stderr",
        "  --debug enables the Ratatui dashboard with charts",
        "  --profiling enables detailed CUDA timing buckets",
        "  default sampling is temperature 0.7 with top_p 0.9",
        "  use --temperature 0 --top-p 1 for greedy parity runs",
        "  NERVA_COLOR=never|ansi|truecolor|always controls log color",
    ]
}

fn serve_usage_lines() -> &'static [&'static str] {
    &[
        "usage:",
        "  nerva serve -m model [--host 127.0.0.1] [--port 8000] [-c context] [-o output] [--max-concurrent-requests count] [--workers count] [--max-blocking-threads count] [--api-key key] [-rt|--rt] [--rt-mode auto|shadow|sparse] [--rt-pages count|--rt-far-pages count] [--rt-page-tokens tokens] [--rt-local-window tokens] [--rt-sink-tokens tokens] [--profiling]",
        "",
        "cargo dev equivalent:",
        "  cargo run -p nerva -- serve -m model [--host 127.0.0.1] [--port 8000] [-c context] [-o output] [--max-concurrent-requests count] [--workers count] [--max-blocking-threads count] [--api-key key] [-rt|--rt] [--rt-mode auto|shadow|sparse] [--rt-pages count|--rt-far-pages count] [--rt-page-tokens tokens] [--rt-local-window tokens] [--rt-sink-tokens tokens] [--profiling]",
    ]
}

#[cfg(test)]
mod tests {
    use super::{is_help_request, is_serve_command, serve_usage_lines, usage_lines};

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn recognizes_serve_command_and_alias() {
        assert!(is_serve_command("serve"));
        assert!(is_serve_command("server"));
        assert!(!is_serve_command("generate"));
    }

    #[test]
    fn recognizes_serve_help_without_treating_it_as_an_error() {
        assert!(is_help_request(&args(&["--help"])));
        assert!(is_help_request(&args(&["-m", "qwen3-8b", "-h"])));
        assert!(!is_help_request(&args(&["-m", "qwen3-8b"])));
    }

    #[test]
    fn usage_mentions_direct_serve_command() {
        assert!(
            usage_lines()
                .iter()
                .any(|line| line.contains("nerva serve -m model"))
        );
        assert!(
            serve_usage_lines()
                .iter()
                .any(|line| line.contains("nerva serve -m model"))
        );
    }

    #[test]
    fn usage_mentions_direct_run_command() {
        assert!(
            usage_lines()
                .iter()
                .any(|line| line.contains("nerva run -m model"))
        );
        assert!(
            usage_lines()
                .iter()
                .any(|line| line.contains("run, generate, chat, ask"))
        );
    }
}
