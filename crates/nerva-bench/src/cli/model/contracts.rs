use std::process::ExitCode;

pub(crate) fn run_kernel_contracts() -> ExitCode {
    match nerva_kernel_contracts::registry::probe::kernel_registry_probe() {
        Ok(summary) => {
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("kernel contract probe failed: {err:?}");
            ExitCode::from(1)
        }
    }
}
