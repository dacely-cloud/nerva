use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

pub(crate) fn run_measurement_table_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_measurement_table_probe()
        .map_err(|err| format!("measurement table probe failed: {err:?}"))?;
    Ok(summary.to_json())
}

pub(crate) fn run_measured_planner_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_measured_planner_probe()
        .map_err(|err| format!("measured planner probe failed: {err:?}"))?;
    Ok(summary.to_json())
}
