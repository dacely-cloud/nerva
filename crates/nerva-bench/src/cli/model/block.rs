use std::process::ExitCode;

pub(crate) fn run_reference_block() -> ExitCode {
    match nerva_model::reference::smoke::run::reference_block_smoke() {
        Ok(summary) => {
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("reference block failed: {err:?}");
            ExitCode::from(1)
        }
    }
}

pub(crate) fn run_safetensors_block() -> ExitCode {
    match nerva_model::precision::file_smoke::run::precision_block_from_safetensors_smoke() {
        Ok(summary) => {
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("safetensors precision block failed: {err:?}");
            ExitCode::from(1)
        }
    }
}
