use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

pub(crate) fn run_transaction_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_execution_transaction_probe()
        .map_err(|err| format!("execution transaction probe failed: {err:?}"))?;
    Ok(summary.to_json())
}
