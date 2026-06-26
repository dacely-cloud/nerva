use std::process::ExitCode;

use crate::cli::exit;
use crate::parse::{parse_optional_u64, parse_optional_usize};
use crate::probes::{
    backend, compute, kv, measurements, memory_loop, mgpu, phase, queue, runtime, synthetic, token,
    transaction, transport,
};

pub(crate) fn dispatch(
    command: Option<&str>,
    args: &mut impl Iterator<Item = String>,
) -> Option<ExitCode> {
    match command {
        Some("capabilities") => Some(exit::print_json_result(runtime::run_capabilities())),
        Some("backend-contract") => Some(exit::print_json_result(
            backend::run_backend_contract_probe(),
        )),
        Some("hot-path-guard") => {
            Some(exit::print_json_result(runtime::run_hot_path_guard_probe()))
        }
        Some("security-isolation") => Some(exit::print_json_result(
            runtime::run_security_isolation_probe(),
        )),
        Some("correctness") => Some(exit::print_json_result(
            runtime::run_correctness_validation_probe(),
        )),
        Some("production-invariants") => Some(exit::print_json_result(
            runtime::run_production_invariant_probe(),
        )),
        Some("request-state") => Some(exit::print_json_result(runtime::run_request_state_probe())),
        Some("request-scheduler") => Some(exit::print_json_result(
            runtime::run_request_scheduler_probe(),
        )),
        Some("topology") => Some(exit::print_json_result(runtime::run_topology_probe())),
        Some("synthetic") => Some(run_synthetic_command(args)),
        Some("ledger") => Some(exit::print_json_result(
            synthetic::run_synthetic_ledger_probe(),
        )),
        Some("critical-path") => {
            Some(exit::print_json_result(synthetic::run_critical_path_probe()))
        }
        Some("token-policy") => Some(exit::print_json_result(token::run_token_policy_probe())),
        Some("phase-handoff") => Some(exit::print_json_result(phase::run_phase_handoff_probe())),
        Some("shared-queue") => Some(exit::print_json_result(queue::run_shared_queue_probe())),
        Some("transaction") => Some(exit::print_json_result(transaction::run_transaction_probe())),
        Some("compute-near-data") => Some(exit::print_json_result(
            compute::run_compute_near_data_probe(),
        )),
        Some("measurements") => Some(exit::print_json_result(
            measurements::run_measurement_table_probe(),
        )),
        Some("measured-planner") => Some(exit::print_json_result(
            measurements::run_measured_planner_probe(),
        )),
        Some("memory-loop") => Some(exit::print_json_result(memory_loop::run_memory_loop_probe())),
        Some("kv") => Some(exit::print_json_result(kv::run_kv_probe())),
        Some("tiered-kv") => Some(exit::print_json_result(kv::run_tiered_kv_attention_probe())),
        Some("fabric-topology") => Some(exit::print_json_result(
            transport::run_fabric_topology_probe(),
        )),
        Some("fabric-backends") => Some(exit::print_json_result(
            transport::run_fabric_backend_probe(),
        )),
        Some("dpdk-udp") => Some(exit::print_json_result(transport::run_dpdk_udp_probe())),
        Some("kernel-udp") => Some(exit::print_json_result(transport::run_kernel_udp_probe())),
        Some("kernel-udp-matrix") => Some(exit::print_json_result(
            transport::run_kernel_udp_matrix_probe(),
        )),
        Some("measured-transport") => Some(exit::print_json_result(
            transport::run_measured_transport_selector_probe(),
        )),
        Some("transport-provenance") => Some(exit::print_json_result(
            transport::run_transport_metric_provenance_probe(),
        )),
        Some("transport") => Some(exit::print_json_result(transport::run_transport_probe())),
        Some("transport-contract") => Some(exit::print_json_result(
            transport::run_transport_contract_probe(),
        )),
        Some("transport-matrix") => Some(exit::print_json_result(
            transport::run_transport_matrix_probe(),
        )),
        Some("transport-registration") => Some(exit::print_json_result(
            transport::run_transport_registration_probe(),
        )),
        Some("transport-registration-lifecycle") => Some(exit::print_json_result(
            transport::run_transport_registration_lifecycle_probe(),
        )),
        Some("stage-pipeline") => Some(exit::print_json_result(
            transport::run_stage_pipeline_probe(),
        )),
        Some("multi-gpu") => Some(exit::print_json_result(mgpu::run_multi_gpu_probe())),
        _ => None,
    }
}

fn run_synthetic_command(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let steps = match parse_optional_u64(args.next(), 1024, "steps") {
        Ok(steps) => steps,
        Err(reason) => return exit::parse_error(reason),
    };
    let ring_capacity = match parse_optional_usize(args.next(), 64, "ring_capacity") {
        Ok(ring_capacity) => ring_capacity,
        Err(reason) => return exit::parse_error(reason),
    };
    exit::print_json_result(synthetic::run_synthetic(steps, ring_capacity))
}
