use std::process::Command;

use crate::json::{json_env_string, json_escape, json_string_array};
use crate::probes::runtime;

pub(crate) fn artifact_metadata_json(command: &str, args: &[String]) -> String {
    let capabilities = runtime::run_capabilities().unwrap_or_else(|reason| {
        format!(
            "{{\"status\":\"failed\",\"error\":\"{}\"}}",
            json_escape(&reason)
        )
    });
    format!(
        "{{\"command\":\"{}\",\"args\":{},\"git_commit\":\"{}\",\"package_version\":\"{}\",\"profile\":\"{}\",\"target\":\"{}-{}\",\"rustc_version\":\"{}\",\"cargo_version\":\"{}\",\"rustflags\":{},\"cargo_encoded_rustflags\":{},\"capabilities\":{}}}",
        json_escape(command),
        json_string_array(args),
        json_escape(&current_git_commit()),
        env!("CARGO_PKG_VERSION"),
        build_profile(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        json_escape(&command_version("rustc")),
        json_escape(&command_version("cargo")),
        json_env_string("RUSTFLAGS"),
        json_env_string("CARGO_ENCODED_RUSTFLAGS"),
        capabilities,
    )
}

fn current_git_commit() -> String {
    if let Some(commit) = option_env!("NERVA_GIT_COMMIT") {
        return commit.to_string();
    }
    let Ok(output) = Command::new("git").args(["rev-parse", "HEAD"]).output() else {
        return "unknown".to_string();
    };
    if !output.status.success() {
        return "unknown".to_string();
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string())
}

fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn command_version(command: &str) -> String {
    let Ok(output) = Command::new(command).arg("--version").output() else {
        return "unknown".to_string();
    };
    if !output.status.success() {
        return "unknown".to_string();
    }
    String::from_utf8(output.stdout)
        .ok()
        .and_then(|stdout| stdout.lines().next().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}
