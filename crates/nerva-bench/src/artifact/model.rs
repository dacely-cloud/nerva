use crate::cli::model::{causal_lm, causal_lm_cuda, causal_lm_cuda_session};
use crate::parse::parse_optional_usize;

pub(crate) fn run_model_artifact(command: &str, args: &[String]) -> Option<Result<String, String>> {
    match command {
        "hf-decode" => Some(run_hf_decode(args)),
        "hf-cuda-decode" => Some(run_hf_cuda_decode(args)),
        "hf-cuda-decode-device-only" => Some(run_hf_cuda_device_only_decode(args)),
        "hf-cuda-decode-device-session" => Some(run_hf_cuda_device_session_decode(args)),
        _ => None,
    }
}

fn run_hf_cuda_device_only_decode(args: &[String]) -> Result<String, String> {
    let steps = parse_optional_usize(args.get(2).cloned(), 8, "steps")?;
    causal_lm_cuda::hf_causal_lm_cuda_device_only_decode_input_json(
        args.first().cloned(),
        args.get(1).cloned(),
        steps,
    )
}

fn run_hf_cuda_device_session_decode(args: &[String]) -> Result<String, String> {
    let max_context = parse_optional_usize(args.get(1).cloned(), 8, "max_context_tokens")?;
    let steps = parse_optional_usize(args.get(2).cloned(), 1, "steps")?;
    causal_lm_cuda_session::hf_causal_lm_cuda_device_session_json(
        args.first().cloned(),
        max_context,
        steps,
        args.iter().skip(3).cloned().collect(),
    )
}

fn run_hf_decode(args: &[String]) -> Result<String, String> {
    let steps = parse_optional_usize(args.get(2).cloned(), 8, "steps")?;
    causal_lm::hf_causal_lm_decode_input_json(args.first().cloned(), args.get(1).cloned(), steps)
}

fn run_hf_cuda_decode(args: &[String]) -> Result<String, String> {
    let steps = parse_optional_usize(args.get(2).cloned(), 8, "steps")?;
    causal_lm_cuda::hf_causal_lm_cuda_decode_input_json(
        args.first().cloned(),
        args.get(1).cloned(),
        steps,
    )
}
