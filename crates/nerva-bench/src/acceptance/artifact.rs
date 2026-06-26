use crate::acceptance::report::AcceptanceReport;
use crate::artifact::run::run_artifact;

pub(crate) fn push_artifact_reproducibility(report: &mut AcceptanceReport) {
    match run_artifact(
        Some("synthetic".to_string()),
        vec!["2".to_string(), "4".to_string()],
    ) {
        Ok(artifact) => report.push(
            "benchmark_artifact_reproducibility",
            artifact.contains("\"artifact_schema\":\"nerva-bench-v1\"")
                && artifact.contains("\"metadata\"")
                && artifact.contains("\"summary\"")
                && artifact.contains("\"command\":\"synthetic\"")
                && artifact.contains("\"args\":[\"2\",\"4\"]")
                && artifact.contains("\"command_line\":[\"cargo\",\"run\",\"-p\",\"nerva-bench\"")
                && artifact.contains("\"cwd\"")
                && artifact.contains("\"git_commit\"")
                && artifact.contains("\"package_version\"")
                && artifact.contains("\"profile\"")
                && artifact.contains("\"target\"")
                && artifact.contains("\"rustc_version\"")
                && artifact.contains("\"cargo_version\"")
                && artifact.contains("\"environment\"")
                && artifact.contains("\"CUDA_VISIBLE_DEVICES\"")
                && artifact.contains("\"HIP_VISIBLE_DEVICES\"")
                && artifact.contains("\"capabilities\"")
                && artifact.contains("\"kernel_release\"")
                && artifact.contains("\"topology\"")
                && artifact.contains("\"observed_token_hash\"")
                && artifact.contains("\"device_timeline_idle_ns\":0"),
            format!(
                "schema={} command={} args={} command_line={} environment={} capabilities={} summary_hash={} idle_zero={}",
                artifact.contains("\"artifact_schema\":\"nerva-bench-v1\""),
                artifact.contains("\"command\":\"synthetic\""),
                artifact.contains("\"args\":[\"2\",\"4\"]"),
                artifact.contains("\"command_line\""),
                artifact.contains("\"environment\""),
                artifact.contains("\"capabilities\""),
                artifact.contains("\"observed_token_hash\""),
                artifact.contains("\"device_timeline_idle_ns\":0"),
            ),
        ),
        Err(err) => report.push(
            "benchmark_artifact_reproducibility",
            false,
            format!("artifact reproducibility probe failed: {err}"),
        ),
    }
}
