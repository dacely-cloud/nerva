use std::process::ExitCode;

pub(crate) fn print_json_result(result: Result<String, String>) -> ExitCode {
    match result {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(reason) => {
            eprintln!("{reason}");
            ExitCode::from(1)
        }
    }
}

pub(crate) fn parse_error(reason: String) -> ExitCode {
    eprintln!("{reason}");
    ExitCode::from(2)
}
