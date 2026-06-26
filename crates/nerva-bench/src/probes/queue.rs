use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

pub(crate) fn run_shared_queue_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_shared_work_queue_probe()
        .map_err(|err| format!("shared queue probe failed: {err:?}"))?;
    Ok(summary.to_json())
}
