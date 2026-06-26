use std::process::ExitCode;

pub(crate) fn run_attention() -> ExitCode {
    match nerva_model::attention::smoke::blockwise_attention_smoke() {
        Ok(summary) => {
            println!("{}", summary.to_json());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("blockwise attention failed: {err:?}");
            ExitCode::from(1)
        }
    }
}
