use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

pub(crate) fn run_token_policy_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_token_policy_probe()
        .map_err(|err| format!("token policy probe failed: {err:?}"))?;
    Ok(summary.to_json())
}
