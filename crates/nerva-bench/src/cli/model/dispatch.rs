use std::process::ExitCode;

use crate::cli::model::{attention, block, contracts, parity, precision, tiny, warm};

pub(crate) fn dispatch(
    command: Option<&str>,
    args: &mut impl Iterator<Item = String>,
) -> Option<ExitCode> {
    match command {
        Some("block") => Some(block::run_reference_block()),
        Some("precision") => Some(precision::run_precision_block()),
        Some("safetensors-block") => Some(block::run_safetensors_block()),
        Some("model") => Some(tiny::run_tiny_model(args)),
        Some("precision-model") => Some(precision::run_tiny_precision_model(args)),
        Some("vllm-parity") => Some(parity::run_vllm_parity(args)),
        Some("attention") => Some(attention::run_attention()),
        Some("warm") => Some(warm::run_warm_compute()),
        Some("contracts") => Some(contracts::run_kernel_contracts()),
        _ => None,
    }
}
