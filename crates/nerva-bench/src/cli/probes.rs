use std::process::ExitCode;

use crate::cli::exit;
use crate::parse::{parse_optional_u64, parse_optional_usize};
use crate::probes::{
    run_capabilities, run_kv_probe, run_synthetic, run_synthetic_ledger_probe, run_topology_probe,
    run_transport_matrix_probe, run_transport_probe,
};

pub(crate) fn dispatch(
    command: Option<&str>,
    args: &mut impl Iterator<Item = String>,
) -> Option<ExitCode> {
    match command {
        Some("capabilities") => Some(exit::print_json_result(run_capabilities())),
        Some("topology") => Some(exit::print_json_result(run_topology_probe())),
        Some("synthetic") => Some(run_synthetic_command(args)),
        Some("ledger") => Some(exit::print_json_result(run_synthetic_ledger_probe())),
        Some("kv") => Some(exit::print_json_result(run_kv_probe())),
        Some("transport") => Some(exit::print_json_result(run_transport_probe())),
        Some("transport-matrix") => Some(exit::print_json_result(run_transport_matrix_probe())),
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
    exit::print_json_result(run_synthetic(steps, ring_capacity))
}
