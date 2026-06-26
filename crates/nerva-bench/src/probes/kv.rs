use nerva_runtime::engine::kv_attention::config::TieredKvAttentionProbeConfig;
use nerva_runtime::engine::kv_probe::config::KvResidencyProbeConfig;
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

pub(crate) fn run_kv_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_kv_residency_probe(KvResidencyProbeConfig::default())
        .map_err(|err| format!("KV residency probe failed: {err:?}"))?;
    Ok(summary.to_json())
}

pub(crate) fn run_tiered_kv_attention_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_tiered_kv_attention_probe(TieredKvAttentionProbeConfig::default())
        .map_err(|err| format!("tiered KV attention probe failed: {err:?}"))?;
    Ok(summary.to_json())
}
