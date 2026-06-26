use std::process::ExitCode;

pub(crate) fn run_warm_compute() -> ExitCode {
    match nerva_model::warm_compute::probe::run::warm_compute_probe() {
        Ok(summary) => {
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("warm compute probe failed: {err:?}");
            ExitCode::from(1)
        }
    }
}
