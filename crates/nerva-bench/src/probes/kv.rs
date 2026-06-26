use nerva_runtime::engine::kv_probe::KvResidencyProbeConfig;
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

pub(crate) fn run_kv_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_kv_residency_probe(KvResidencyProbeConfig::default())
        .map_err(|err| format!("KV residency probe failed: {err:?}"))?;
    Ok(summary.to_json())
}
