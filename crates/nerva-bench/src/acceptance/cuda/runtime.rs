use crate::acceptance::report::AcceptanceReport;

pub(crate) fn push_smoke(report: &mut AcceptanceReport) {
    let cuda_smoke = nerva_runtime::capabilities::discovery::cuda_smoke();
    let cuda_smoke_passed = format!("{:?}", cuda_smoke.status) == "Ok"
        && cuda_smoke.kernel_value == Some(0x4e45_5256)
        && cuda_smoke.device_free_memory_bytes.unwrap_or(0) > 0
        && cuda_smoke.hot_path_allocations == 0;
    report.push(
        "cuda_runtime_smoke",
        cuda_smoke_passed,
        format!(
            "status={:?} gpu={} cc={}.{} memory_bytes={} free_memory_bytes={} pci_bus_id={} value={} hot_path_allocations={} error={}",
            cuda_smoke.status,
            cuda_smoke.gpu_name.as_deref().unwrap_or("none"),
            cuda_smoke
                .compute_capability_major
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            cuda_smoke
                .compute_capability_minor
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            cuda_smoke
                .device_total_memory_bytes
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            cuda_smoke
                .device_free_memory_bytes
                .map_or_else(|| "none".to_string(), |value| value.to_string()),
            cuda_smoke.pci_bus_id.as_deref().unwrap_or("none"),
            cuda_smoke
                .kernel_value
                .map_or_else(|| "none".to_string(), |value| format!("0x{value:08x}")),
            cuda_smoke.hot_path_allocations,
            cuda_smoke.error.as_deref().unwrap_or("none"),
        ),
    );
}
