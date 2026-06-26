use crate::artifact::{dispatch, metadata};

pub(crate) fn run_artifact(command: Option<String>, args: Vec<String>) -> Result<String, String> {
    let command = command.ok_or_else(|| "artifact requires a probe name".to_string())?;
    let summary = dispatch::run_artifact_probe(&command, &args)?;
    Ok(format!(
        "{{\"status\":\"ok\",\"artifact_schema\":\"nerva-bench-v1\",\"metadata\":{},\"summary\":{}}}",
        metadata::artifact_metadata_json(&command, &args),
        summary
    ))
}
