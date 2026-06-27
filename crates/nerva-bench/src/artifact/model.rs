use crate::cli::model::{
    causal_lm, causal_lm_cuda, causal_lm_cuda_generate, causal_lm_cuda_session,
    causal_lm_cuda_session_loop, causal_lm_cuda_session_stream,
};
use crate::parse::parse_optional_usize;

pub(crate) fn run_model_artifact(command: &str, args: &[String]) -> Option<Result<String, String>> {
    match command {
        "hf-decode" => Some(run_hf_decode(args)),
        "hf-cuda-decode" => Some(run_hf_cuda_decode(args)),
        "hf-cuda-decode-device-only" => Some(run_hf_cuda_device_only_decode(args)),
        "hf-cuda-decode-device-session" => Some(run_hf_cuda_device_session_decode(args)),
        "hf-cuda-decode-device-session-loop" => Some(run_hf_cuda_device_session_loop(args)),
        "hf-cuda-decode-device-session-stream" => Some(run_hf_cuda_device_session_stream(args)),
        "hf-cuda-generate" => Some(run_hf_cuda_generate(args)),
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

fn run_hf_cuda_device_session_loop(args: &[String]) -> Result<String, String> {
    let max_context = parse_optional_usize(args.get(1).cloned(), 8, "max_context_tokens")?;
    let chunk_steps = parse_optional_usize(args.get(2).cloned(), 1, "chunk_steps")?;
    let chunks = parse_optional_usize(args.get(3).cloned(), 1, "chunks")?;
    causal_lm_cuda_session_loop::hf_causal_lm_cuda_device_session_loop_json(
        args.first().cloned(),
        max_context,
        chunk_steps,
        chunks,
        args.get(4).cloned(),
    )
}

fn run_hf_cuda_device_session_stream(args: &[String]) -> Result<String, String> {
    let max_context = parse_optional_usize(args.get(1).cloned(), 8, "max_context_tokens")?;
    let chunk_steps = parse_optional_usize(args.get(2).cloned(), 1, "chunk_steps")?;
    let chunks = parse_optional_usize(args.get(3).cloned(), 1, "chunks")?;
    let capacity = parse_optional_usize(args.get(4).cloned(), 2, "queue_capacity")?;
    causal_lm_cuda_session_stream::hf_causal_lm_cuda_device_session_stream_json(
        args.first().cloned(),
        max_context,
        chunk_steps,
        chunks,
        capacity,
        args.get(5).cloned(),
    )
}

fn run_hf_cuda_generate(args: &[String]) -> Result<String, String> {
    let max_context = parse_optional_usize(args.get(1).cloned(), 8, "max_context_tokens")?;
    let max_new_tokens = parse_optional_usize(args.get(2).cloned(), 16, "max_new_tokens")?;
    let capacity = parse_optional_usize(args.get(3).cloned(), 64, "queue_capacity")?;
    causal_lm_cuda_generate::hf_causal_lm_cuda_generate_json(
        args.first().cloned(),
        max_context,
        max_new_tokens,
        capacity,
        args.get(4).cloned(),
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
