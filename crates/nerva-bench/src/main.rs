#[cfg(not(target_os = "linux"))]
compile_error!("NERVA currently supports Linux only.");

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("smoke") => {
            let summary = nerva_runtime::cuda_smoke();
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("usage: cargo run -p nerva-bench -- smoke");
            ExitCode::from(2)
        }
    }
}
