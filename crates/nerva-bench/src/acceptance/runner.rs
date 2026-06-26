use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

use crate::acceptance::report::AcceptanceReport;
use crate::acceptance::{
    artifact, audit, backend, correctness, cuda, environment, execution, measurements, memory_loop,
    mgpu, model, phase, production, queue, request, resident_weights, runtime_checks, security,
    token, transport,
};

pub(crate) fn build_acceptance_report() -> Result<AcceptanceReport, String> {
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let mut report = AcceptanceReport::default();

    environment::push_capability_provenance(&mut report, &runtime);
    artifact::push_artifact_reproducibility(&mut report);
    report.push_audit_result("vllm_rvllm_audit", audit::audit_acceptance());
    backend::push_backend_contract(&mut report, &runtime);

    cuda::runtime::push_smoke(&mut report);
    cuda::backend::push_backend_contract(&mut report);
    cuda::graph::push_transaction(&mut report);
    cuda::sampler::push_device_sampler(&mut report);

    runtime_checks::push_static_arenas(&mut report, &runtime);
    runtime_checks::push_hot_path_guard(&mut report, &runtime);
    security::push_security_isolation(&mut report, &runtime);
    correctness::push_correctness_validation(&mut report, &runtime);
    production::push_production_invariants(&mut report, &runtime);
    request::push_request_state(&mut report);
    environment::push_topology_snapshot(&mut report, &runtime);
    runtime_checks::push_synthetic_decode(&mut report, &runtime);
    runtime_checks::push_critical_path(&mut report, &runtime);
    token::push_token_policy(&mut report, &runtime);
    phase::push_phase_handoff(&mut report, &runtime);
    queue::push_shared_queue(&mut report, &runtime);
    execution::push_transaction_planner(&mut report, &runtime);
    execution::push_compute_near_data(&mut report, &runtime);
    measurements::push_measurement_table(&mut report, &runtime);
    measurements::push_measured_planner(&mut report, &runtime);
    memory_loop::push_memory_fabric_loop(&mut report, &runtime);

    model::reference::push_reference_block(&mut report);
    model::precision::push_precision_and_cuda_blocks(&mut report);
    model::tiny::push_tiny_model_and_cuda_decode(&mut report);
    model::prompt::push_prompt_model(&mut report);
    model::file_checks::push_manifest_and_file_checks(&mut report);
    model::attention::push_tiered_attention_and_cuda(&mut report);
    model::warm::push_warm_compute(&mut report);
    model::contracts::push_kernel_contracts(&mut report);

    transport::kv::push_kv_residency(&mut report, &runtime);
    transport::kv::push_tiered_kv_attention(&mut report, &runtime);
    transport::fabric::push_fabric_topology(&mut report, &runtime);
    transport::fabric::push_fabric_backends(&mut report, &runtime);
    transport::dpdk_udp::push_dpdk_udp_protocol(&mut report, &runtime);
    transport::kernel_udp::push_kernel_udp_baseline(&mut report, &runtime);
    transport::kernel_udp::push_kernel_udp_matrix(&mut report, &runtime);
    transport::measured::push_measured_transport_selector(&mut report, &runtime);
    transport::provenance::push_transport_metric_provenance(&mut report, &runtime);
    transport::path::push_transport_path(&mut report, &runtime);
    transport::contract::push_transport_contract(&mut report, &runtime);
    transport::matrix::push_transport_matrix(&mut report, &runtime);
    transport::registration::push_transport_registration(&mut report, &runtime);
    transport::registration::push_transport_registration_lifecycle(&mut report, &runtime);
    transport::stage::push_stage_pipeline(&mut report, &runtime);
    mgpu::push_multi_gpu_node(&mut report, &runtime);

    match resident_weights::resident_weight_execution_acceptance(&runtime) {
        Ok((passed, details)) => report.push("resident_weight_execution", passed, details),
        Err(err) => report.push("resident_weight_execution", false, err),
    }

    Ok(report)
}

pub(crate) fn run_acceptance_probe() -> Result<String, String> {
    build_acceptance_report().map(|report| report.to_json())
}
