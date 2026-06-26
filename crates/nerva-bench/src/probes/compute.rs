use nerva_runtime::engine::compute_near_data::config::ComputeNearDataProbeConfig;
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

pub(crate) fn run_compute_near_data_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_compute_near_data_probe(ComputeNearDataProbeConfig::default())
        .map_err(|err| format!("compute-near-data probe failed: {err:?}"))?;
    Ok(summary.to_json())
}
