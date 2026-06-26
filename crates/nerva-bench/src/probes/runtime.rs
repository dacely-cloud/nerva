use nerva_runtime::engine::residency::ResidencyBudget;
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

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

pub(crate) fn run_hot_path_guard_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    runtime
        .run_hot_path_guard_probe(ResidencyBudget::new(1024, 2048, 4096))
        .map(|summary| summary.to_json())
        .map_err(|err| format!("hot-path guard probe failed: {err:?}"))
}

pub(crate) fn run_security_isolation_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    runtime
        .run_security_isolation_probe()
        .map(|summary| summary.to_json())
        .map_err(|err| format!("security isolation probe failed: {err:?}"))
}

pub(crate) fn run_correctness_validation_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    runtime
        .run_correctness_validation_probe()
        .map(|summary| summary.to_json())
        .map_err(|err| format!("correctness validation probe failed: {err:?}"))
}

pub(crate) fn run_production_invariant_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    runtime
        .run_production_invariant_probe()
        .map(|summary| summary.to_json())
        .map_err(|err| format!("production invariant probe failed: {err:?}"))
}
