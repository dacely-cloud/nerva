use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

mod audit;
mod cuda;
mod environment;
mod files;
mod manifest;
mod model;
mod report;
mod resident_weights;
mod runtime_checks;
mod transport;
mod vllm;

use crate::acceptance::report::AcceptanceReport;

pub(crate) fn build_acceptance_report() -> Result<AcceptanceReport, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let mut report = AcceptanceReport::default();

    environment::push_capability_provenance(&mut report, &runtime);
    report.push_audit_result("vllm_rvllm_audit", audit::audit_acceptance());

    cuda::runtime::push_smoke(&mut report);
    cuda::graph::push_transaction(&mut report);
    cuda::sampler::push_device_sampler(&mut report);

    runtime_checks::push_static_arenas(&mut report, &runtime);
    environment::push_topology_snapshot(&mut report, &runtime);
    runtime_checks::push_synthetic_decode(&mut report, &runtime);

    model::push_reference_block(&mut report);
    model::push_precision_and_cuda_blocks(&mut report);
    model::push_tiny_model_and_cuda_decode(&mut report);
    model::push_manifest_and_file_checks(&mut report);
    model::push_tiered_attention_and_cuda(&mut report);
    model::push_warm_compute(&mut report);
    model::push_kernel_contracts(&mut report);

    transport::kv::push_kv_residency(&mut report, &runtime);
    transport::path::push_transport_path(&mut report, &runtime);
    transport::matrix::push_transport_matrix(&mut report, &runtime);
    transport::stage::push_stage_pipeline(&mut report, &runtime);

    match resident_weights::resident_weight_execution_acceptance(&runtime) {
        Ok((passed, details)) => report.push("resident_weight_execution", passed, details),
        Err(err) => report.push("resident_weight_execution", false, err),
    }

    Ok(report)
}

pub(crate) fn run_acceptance_probe() -> Result<String, String> {
    build_acceptance_report().map(|report| report.to_json())
}
