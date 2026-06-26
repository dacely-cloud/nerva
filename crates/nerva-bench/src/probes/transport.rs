use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};
use nerva_runtime::transport::stage::config::StagePipelineConfig;

pub(crate) fn run_transport_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_transport_path_probe()
        .map_err(|err| format!("transport path probe failed: {err:?}"))?;
    Ok(summary.to_json())
}

pub(crate) fn run_fabric_topology_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    Ok(runtime.run_fabric_topology_probe().to_json())
}

pub(crate) fn run_fabric_backend_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    Ok(runtime.run_fabric_backend_probe().to_json())
}

pub(crate) fn run_transport_matrix_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_transport_capability_matrix_probe()
        .map_err(|err| format!("transport capability matrix probe failed: {err:?}"))?;
    Ok(summary.to_json())
}

pub(crate) fn run_stage_pipeline_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_stage_pipeline_probe(StagePipelineConfig::reference_decode())
        .map_err(|err| format!("stage pipeline probe failed: {err:?}"))?;
    Ok(summary.to_json())
}
