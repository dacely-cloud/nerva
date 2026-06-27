use std::process::ExitCode;

use nerva_core::types::id::token::TokenId;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_runtime::engine::hf_cuda_decode::file_backed::session_loop::{
    HfCudaDeviceSessionLoopOutput, run_hf_causal_lm_cuda_shard_backed_device_session_loop,
};
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

use crate::cli::exit;
use crate::cli::model::causal_lm_cuda_session::parse_prompt;
use crate::cli::model::causal_lm_text::generated_text_json;
use crate::json::json_escape;
use crate::parse::parse_optional_usize;

pub(crate) fn run_hf_causal_lm_cuda_device_session_loop(
    args: &mut impl Iterator<Item = String>,
) -> ExitCode {
    let path = args.next();
    let max_context = match parse_optional_usize(args.next(), 8, "max_context_tokens") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let chunk_steps = match parse_optional_usize(args.next(), 1, "chunk_steps") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let chunks = match parse_optional_usize(args.next(), 1, "chunks") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    exit::print_json_result(hf_causal_lm_cuda_device_session_loop_json(
        path,
        max_context,
        chunk_steps,
        chunks,
        args.next(),
    ))
}

pub(crate) fn hf_causal_lm_cuda_device_session_loop_json(
    path: Option<String>,
    max_context_tokens: usize,
    chunk_steps: usize,
    chunks: usize,
    prompt_spec: Option<String>,
) -> Result<String, String> {
    let path = path
        .ok_or_else(|| "hf-cuda-decode-device-session-loop requires checkpoint_dir".to_string())?;
    let prompt = prompt_spec.unwrap_or_else(|| "0".to_string());
    let prompt_ids = parse_prompt(&prompt)?;
    let token_ids = prompt_ids.iter().copied().map(TokenId).collect::<Vec<_>>();
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let output = run_hf_causal_lm_cuda_shard_backed_device_session_loop(
        &runtime,
        &path,
        &token_ids,
        max_context_tokens,
        chunk_steps,
        chunks,
        None,
    )
    .map_err(|err| format!("HF CUDA session loop failed: {err:?}"))?;
    output_json(&path, &prompt, &prompt_ids, chunk_steps, chunks, &output)
}

fn output_json(
    path: &str,
    prompt: &str,
    prompt_ids: &[u32],
    chunk_steps: usize,
    chunks_requested: usize,
    output: &HfCudaDeviceSessionLoopOutput,
) -> Result<String, String> {
    let dtype = nerva_model::precision::bits::dtype_label(output.dtype)
        .map_err(|err| format!("HF causal LM dtype failed: {err:?}"))?;
    let generated_text = generated_text_json(path, &output.tokens)?;
    Ok(format!(
        "{{\"status\":\"{}\",\"backend\":\"cuda\",\"mode\":\"device_session_loop\",\"path\":\"{}\",\"prompt\":\"{}\",\"prompt_token_ids\":{},\"chunk_steps\":{},\"chunks_requested\":{},\"chunks_observed\":{},\"dtype\":\"{}\",\"layers\":{},\"hidden\":{},\"vocab_size\":{},\"manifest_entries\":{},\"shard_plan_entries\":{},\"tensors_loaded\":{},\"bytes_loaded\":{},\"data_hash\":{},\"data_hash_available\":{},\"generated_text\":{},\"create\":{},\"start\":{},\"chunks\":{}}}",
        if output
            .chunks
            .iter()
            .all(|chunk| chunk.summary.status == SmokeStatus::Ok)
        {
            "ok"
        } else {
            "failed"
        },
        json_escape(path),
        json_escape(prompt),
        u32s_json(prompt_ids),
        chunk_steps,
        chunks_requested,
        output.chunks.len(),
        dtype,
        output.metadata.num_hidden_layers,
        output.metadata.hidden_size,
        output.metadata.vocab_size,
        output.manifest_entries,
        output.shard_plan_entries,
        output.tensors_loaded,
        output.bytes_loaded,
        output.data_hash,
        output.data_hash_available,
        generated_text,
        output.create.to_json(),
        output.start.to_json(),
        chunks_json(path, output)?,
    ))
}

fn chunks_json(path: &str, output: &HfCudaDeviceSessionLoopOutput) -> Result<String, String> {
    let mut out = String::from("[");
    for (index, chunk) in output.chunks.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let text = generated_text_json(path, &chunk.summary.tokens)?;
        out.push_str(&format!(
            "{{\"chunk_index\":{},\"requested_steps\":{},\"generated_text\":{},\"summary\":{}}}",
            chunk.chunk_index,
            chunk.requested_steps,
            text,
            chunk.summary.to_json(),
        ));
    }
    out.push(']');
    Ok(out)
}

fn u32s_json(values: &[u32]) -> String {
    let mut out = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&value.to_string());
    }
    out.push(']');
    out
}
