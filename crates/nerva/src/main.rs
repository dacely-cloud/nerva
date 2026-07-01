#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

mod cli;
mod json;
mod openai;

fn main() -> std::process::ExitCode {
    cli::run()
}
