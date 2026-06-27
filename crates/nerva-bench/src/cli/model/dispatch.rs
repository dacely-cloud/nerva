use std::process::ExitCode;

use crate::cli::model::{
    attention, block, causal_lm, causal_lm_cuda, causal_lm_cuda_session,
    causal_lm_cuda_session_loop, causal_lm_cuda_session_stream, contracts, parity, precision,
    prompt, tiny, warm,
};

pub(crate) fn dispatch(
    command: Option<&str>,
    args: &mut impl Iterator<Item = String>,
) -> Option<ExitCode> {
    match command {
        Some("block") => Some(block::run_reference_block()),
        Some("precision") => Some(precision::run_precision_block()),
        Some("safetensors-block") => Some(block::run_safetensors_block()),
        Some("model") => Some(tiny::run_tiny_model(args)),
        Some("prompt-model") => Some(prompt::run_prompt_model(args)),
        Some("precision-model") => Some(precision::run_tiny_precision_model(args)),
        Some("hf-decode") => Some(causal_lm::run_hf_causal_lm_decode(args)),
        Some("hf-cuda-decode") => Some(causal_lm_cuda::run_hf_causal_lm_cuda_decode(args)),
        Some("hf-cuda-decode-device-only") => Some(
            causal_lm_cuda::run_hf_causal_lm_cuda_device_only_decode(args),
        ),
        Some("hf-cuda-decode-device-session") => {
            Some(causal_lm_cuda_session::run_hf_causal_lm_cuda_device_session_decode(args))
        }
        Some("hf-cuda-decode-device-session-loop") => {
            Some(causal_lm_cuda_session_loop::run_hf_causal_lm_cuda_device_session_loop(args))
        }
        Some("hf-cuda-decode-device-session-stream") => {
            Some(causal_lm_cuda_session_stream::run_hf_causal_lm_cuda_device_session_stream(args))
        }
        Some("vllm-parity") => Some(parity::run_vllm_parity(args)),
        Some("attention") => Some(attention::run_attention()),
        Some("warm") => Some(warm::run_warm_compute()),
        Some("contracts") => Some(contracts::run_kernel_contracts()),
        _ => None,
    }
}
