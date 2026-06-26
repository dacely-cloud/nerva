use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};
use nerva_runtime::transport::dpdk_udp::config::DpdkUdpProbeConfig;
use nerva_runtime::transport::kernel_udp::config::KernelUdpProbeConfig;
use nerva_runtime::transport::stage::config::StagePipelineConfig;

pub(crate) fn run_transport_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_transport_path_probe()
        .map_err(|err| format!("transport path probe failed: {err:?}"))?;
    Ok(summary.to_json())
}

pub(crate) fn run_transport_contract_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_transport_contract_probe()
        .map_err(|err| format!("transport contract probe failed: {err:?}"))?;
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

pub(crate) fn run_dpdk_udp_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_dpdk_udp_protocol_probe(DpdkUdpProbeConfig::reference_decode_activation())
        .map_err(|err| format!("DPDK UDP protocol probe failed: {err:?}"))?;
    Ok(summary.to_json())
}

pub(crate) fn run_kernel_udp_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_kernel_udp_baseline_probe(KernelUdpProbeConfig::reference_decode_activation())
        .map_err(|err| format!("kernel UDP baseline probe failed: {err:?}"))?;
    Ok(summary.to_json())
}

pub(crate) fn run_kernel_udp_matrix_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_kernel_udp_baseline_matrix_probe()
        .map_err(|err| format!("kernel UDP baseline matrix probe failed: {err:?}"))?;
    Ok(summary.to_json())
}

pub(crate) fn run_measured_transport_selector_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_measured_transport_selector_probe()
        .map_err(|err| format!("measured transport selector probe failed: {err:?}"))?;
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

pub(crate) fn run_transport_registration_probe() -> Result<String, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let summary = runtime
        .run_transport_registration_probe()
        .map_err(|err| format!("transport registration probe failed: {err:?}"))?;
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
