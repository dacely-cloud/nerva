use std::process::ExitCode;

use crate::cli::exit;
use crate::model_io::config::{run_layout_probe, run_manifest_probe, run_metadata_probe};
use crate::model_io::resident::{
    run_hotset_probe, run_resident_shard_probe, run_resident_weight_probe,
    run_weight_execution_probe,
};
use crate::model_io::safetensors::{run_safetensors_probe, run_safetensors_shard_probe};
use crate::parse::{parse_optional_u64, parse_optional_usize};

pub(crate) fn dispatch(
    command: Option<&str>,
    args: &mut impl Iterator<Item = String>,
) -> Option<ExitCode> {
    match command {
        Some("metadata") => Some(exit::print_json_result(run_metadata_probe(args.next()))),
        Some("layout") => Some(exit::print_json_result(run_layout_probe(args.next()))),
        Some("manifest") => Some(exit::print_json_result(run_manifest_probe(args.next()))),
        Some("safetensors") => Some(exit::print_json_result(run_safetensors_probe(
            args.next(),
            args.next(),
        ))),
        Some("safetensors-shards") => Some(exit::print_json_result(run_safetensors_shard_probe(
            args.next(),
            args.next(),
            args.next(),
        ))),
        Some("resident-shards") => Some(run_resident_shards(args)),
        Some("resident-weights") => Some(exit::print_json_result(run_resident_weight_probe(
            args.next(),
        ))),
        Some("hotset") => Some(run_hotset(args)),
        Some("weight-exec") => Some(run_weight_execution(args)),
        _ => None,
    }
}

fn run_resident_shards(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let config_path = args.next();
    let index_path = args.next();
    let checkpoint_dir = args.next();
    let max_task_bytes = match parse_optional_usize(args.next(), 16 * 1024 * 1024, "max_task_bytes")
    {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    exit::print_json_result(run_resident_shard_probe(
        config_path,
        index_path,
        checkpoint_dir,
        max_task_bytes,
    ))
}

fn run_hotset(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let config_path = args.next();
    let vram_bytes = match parse_optional_usize(args.next(), 512 * 1024 * 1024, "vram_bytes") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let max_promote_bytes = match parse_optional_usize(args.next(), vram_bytes, "max_promote_bytes")
    {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    exit::print_json_result(run_hotset_probe(config_path, vram_bytes, max_promote_bytes))
}

fn run_weight_execution(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let config_path = args.next();
    let vram_bytes = match parse_optional_usize(args.next(), 512 * 1024 * 1024, "vram_bytes") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let max_promote_bytes = match parse_optional_usize(args.next(), vram_bytes, "max_promote_bytes")
    {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let max_steps = match parse_optional_usize(args.next(), 32, "max_steps") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let compute_capability = match parse_optional_u64(args.next(), 89, "compute_capability") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    exit::print_json_result(run_weight_execution_probe(
        config_path,
        vram_bytes,
        max_promote_bytes,
        max_steps,
        Some(compute_capability as u32),
    ))
}
