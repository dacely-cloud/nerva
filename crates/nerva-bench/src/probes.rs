use nerva_core::types::{RequestId, SequenceId, TokenId};
use nerva_runtime::engine::{
    KvResidencyProbeConfig, Runtime, RuntimeConfig, SyntheticDecodeConfig,
};

pub(crate) fn run_capabilities() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    Ok(runtime.discover_capabilities().to_json())
}

pub(crate) fn run_topology_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    Ok(runtime.discover_topology().to_json())
}

pub(crate) fn run_synthetic(steps: u64, ring_capacity: usize) -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_synthetic_decode(SyntheticDecodeConfig::new(steps, ring_capacity, TokenId(1)))
        .map_err(|err| format!("synthetic decode failed: {err:?}"))?;
    Ok(summary.to_json())
}

pub(crate) fn run_synthetic_ledger_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let mut engine = runtime
        .synthetic_engine(4)
        .map_err(|err| format!("synthetic engine init failed: {err:?}"))?;
    let output = engine
        .launch_device_next(RequestId(1), SequenceId(1), 0, TokenId(1))
        .map_err(|err| format!("synthetic ledger launch failed: {err:?}"))?
        .collect()
        .map_err(|err| format!("synthetic ledger collect failed: {err:?}"))?;
    Ok(output.ledger.to_json())
}

pub(crate) fn run_kv_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_kv_residency_probe(KvResidencyProbeConfig::default())
        .map_err(|err| format!("KV residency probe failed: {err:?}"))?;
    Ok(summary.to_json())
}

pub(crate) fn run_transport_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_transport_path_probe()
        .map_err(|err| format!("transport path probe failed: {err:?}"))?;
    Ok(summary.to_json())
}

pub(crate) fn run_transport_matrix_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_transport_capability_matrix_probe()
        .map_err(|err| format!("transport capability matrix probe failed: {err:?}"))?;
    Ok(summary.to_json())
}
