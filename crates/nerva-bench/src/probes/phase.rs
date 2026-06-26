use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

pub(crate) fn run_phase_handoff_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_phase_handoff_probe()
        .map_err(|err| format!("phase handoff probe failed: {err:?}"))?;
    Ok(summary.to_json())
}
