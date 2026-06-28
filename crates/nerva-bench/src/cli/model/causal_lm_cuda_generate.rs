use std::fs;
use std::process::ExitCode;

use nerva_core::types::id::token::TokenId;
use nerva_runtime::engine::hf_cuda_decode::file_backed::generate::{
    HfCudaDeviceGenerateOutput, run_hf_causal_lm_cuda_shard_backed_device_generate,
};
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

use crate::cli::exit;
use crate::cli::model::causal_lm_cuda_perf::stream_perf_json;
use crate::cli::model::causal_lm_cuda_session_stream::{
    chunks_json, queue_json, records_json, u32s_json,
};
use crate::cli::model::causal_lm_text::generated_text_json;
use crate::json::json_escape;
use crate::parse::{parse_optional_u32, parse_optional_usize};
use nerva_model::hf::tokenizer::encode_text_prompt;

pub(crate) fn run_hf_causal_lm_cuda_generate(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let path = args.next();
    let max_context = match parse_optional_usize(args.next(), 8, "max_context_tokens") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let max_new_tokens = match parse_optional_usize(args.next(), 16, "max_new_tokens") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let queue_capacity = match parse_optional_usize(args.next(), 64, "queue_capacity") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    exit::print_json_result(hf_causal_lm_cuda_generate_json(
        path,
        max_context,
        max_new_tokens,
        queue_capacity,
        args.next(),
        match parse_optional_u32(args.next(), 0, "compute_capability") {
            Ok(0) => None,
            Ok(value) => Some(value),
            Err(reason) => return exit::parse_error(reason),
        },
    ))
}

pub(crate) fn hf_causal_lm_cuda_generate_json(
    path: Option<String>,
    max_context_tokens: usize,
    max_new_tokens: usize,
    queue_capacity: usize,
    prompt_spec: Option<String>,
    compute_capability: Option<u32>,
) -> Result<String, String> {
    let path = path.ok_or_else(|| "hf-cuda-generate requires checkpoint_dir".to_string())?;
    let prompt_spec =
        prompt_spec.ok_or_else(|| "hf-cuda-generate requires prompt_text".to_string())?;
    let prompt = resolve_prompt_text(&prompt_spec)?;
    let encoded = encode_text_prompt(&path, &prompt)?;
    let token_ids = encoded
        .token_ids
        .iter()
        .copied()
        .map(TokenId)
        .collect::<Vec<_>>();
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let output = run_hf_causal_lm_cuda_shard_backed_device_generate(
        &runtime,
        &path,
        &token_ids,
        max_context_tokens,
        max_new_tokens,
        queue_capacity,
        compute_capability,
    )
    .map_err(|err| format!("HF CUDA generate failed: {err:?}"))?;
    generate_json(
        &path,
        &prompt,
        encoded.input_mode,
        &encoded.token_ids,
        &output,
    )
}

fn resolve_prompt_text(prompt_spec: &str) -> Result<String, String> {
    let Some(path) = prompt_spec.strip_prefix('@') else {
        return Ok(prompt_spec.to_string());
    };
    if path.is_empty() {
        return Err("hf-cuda-generate prompt file path is empty".to_string());
    }
    fs::read_to_string(path)
        .map_err(|err| format!("hf-cuda-generate failed to read prompt file {path}: {err}"))
}

pub(crate) fn generate_json(
    path: &str,
    prompt: &str,
    input_mode: &str,
    prompt_ids: &[u32],
    output: &HfCudaDeviceGenerateOutput,
) -> Result<String, String> {
    let stream = &output.stream;
    let dtype = nerva_model::precision::bits::dtype_label(stream.dtype)
        .map_err(|err| format!("HF CUDA generate dtype failed: {err:?}"))?;
    let generated_text = generated_text_json(path, output.tokens())?;
    Ok(format!(
        "{{\"status\":\"ok\",\"backend\":\"cuda\",\"mode\":\"device_generate\",\"path\":\"{}\",\"input_mode\":\"{}\",\"prompt\":\"{}\",\"prompt_token_ids\":{},\"prompt_tokens\":{},\"max_new_tokens\":{},\"tokens\":{},\"generated_text\":{},\"stop_reason\":\"{}\",\"chunks_observed\":{},\"dtype\":\"{}\",\"layers\":{},\"hidden\":{},\"vocab_size\":{},\"manifest_entries\":{},\"shard_plan_entries\":{},\"tensors_loaded\":{},\"bytes_loaded\":{},\"data_hash\":{},\"data_hash_available\":{},\"perf\":{},\"queue\":{},\"create\":{},\"start\":{},\"records\":{},\"chunks\":{}}}",
        json_escape(path),
        input_mode,
        json_escape(prompt),
        u32s_json(prompt_ids),
        prompt_ids.len(),
        output.max_new_tokens,
        token_ids_json(output.tokens()),
        generated_text,
        output.stop_reason().as_str(),
        stream.chunks.len(),
        dtype,
        stream.metadata.num_hidden_layers,
        stream.metadata.hidden_size,
        stream.metadata.vocab_size,
        stream.manifest_entries,
        stream.shard_plan_entries,
        stream.tensors_loaded,
        stream.bytes_loaded,
        stream.data_hash,
        stream.data_hash_available,
        stream_perf_json(stream),
        queue_json(stream),
        stream.create.to_json(),
        stream.start.to_json(),
        records_json(stream),
        chunks_json(path, stream)?,
    ))
}

fn token_ids_json(tokens: &[TokenId]) -> String {
    let values = tokens.iter().map(|token| token.0).collect::<Vec<_>>();
    u32s_json(&values)
}
