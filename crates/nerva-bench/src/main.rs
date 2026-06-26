#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

mod acceptance;
mod artifact;
mod cli;
mod json;
mod model_io;
mod parse;
mod probes;

#[cfg(test)]
mod tests;

fn main() -> std::process::ExitCode {
    cli::run()
}
