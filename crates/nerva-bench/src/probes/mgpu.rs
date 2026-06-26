use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};
use nerva_runtime::mgpu::config::MultiGpuNodeConfig;

pub(crate) fn run_multi_gpu_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_multi_gpu_node_probe(MultiGpuNodeConfig::reference_2080ti_stage())
        .map_err(|err| format!("multi-GPU node probe failed: {err:?}"))?;
    Ok(summary.to_json())
}
