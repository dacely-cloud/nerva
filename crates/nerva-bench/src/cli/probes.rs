use std::process::ExitCode;

use crate::cli::exit;
use crate::parse::{parse_optional_u64, parse_optional_usize};
use crate::probes::{
    kv, memory_loop, mgpu, phase, queue, runtime, synthetic, token, transaction, transport,
};

pub(crate) fn dispatch(
    command: Option<&str>,
    args: &mut impl Iterator<Item = String>,
) -> Option<ExitCode> {
    match command {
        Some("capabilities") => Some(exit::print_json_result(runtime::run_capabilities())),
        Some("topology") => Some(exit::print_json_result(runtime::run_topology_probe())),
        Some("synthetic") => Some(run_synthetic_command(args)),
        Some("ledger") => Some(exit::print_json_result(
            synthetic::run_synthetic_ledger_probe(),
        )),
        Some("token-policy") => Some(exit::print_json_result(token::run_token_policy_probe())),
        Some("phase-handoff") => Some(exit::print_json_result(phase::run_phase_handoff_probe())),
        Some("shared-queue") => Some(exit::print_json_result(queue::run_shared_queue_probe())),
        Some("transaction") => Some(exit::print_json_result(transaction::run_transaction_probe())),
        Some("memory-loop") => Some(exit::print_json_result(memory_loop::run_memory_loop_probe())),
        Some("kv") => Some(exit::print_json_result(kv::run_kv_probe())),
        Some("fabric-topology") => Some(exit::print_json_result(
            transport::run_fabric_topology_probe(),
        )),
        Some("fabric-backends") => Some(exit::print_json_result(
            transport::run_fabric_backend_probe(),
        )),
        Some("transport") => Some(exit::print_json_result(transport::run_transport_probe())),
        Some("transport-matrix") => Some(exit::print_json_result(
            transport::run_transport_matrix_probe(),
        )),
        Some("transport-registration") => Some(exit::print_json_result(
            transport::run_transport_registration_probe(),
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
