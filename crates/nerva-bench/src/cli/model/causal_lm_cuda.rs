use std::process::ExitCode;

use nerva_core::types::id::token::TokenId;
use nerva_model::causal_lm::load_options::HfCausalLmLoadOptions;
use nerva_runtime::engine::hf_cuda_decode::run::{
    run_loaded_hf_causal_lm_cuda_prompt_decode,
    run_loaded_hf_causal_lm_cuda_prompt_decode_device_only,
    run_loaded_hf_causal_lm_cuda_seed_decode,
};
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};
use tokenizers::Tokenizer;

use crate::cli::exit;
use crate::cli::model::causal_lm_cuda_json::{HfCudaDecodeJson, hf_cuda_decode_json};
use crate::cli::model::causal_lm_text::generated_text_json;
use crate::parse::{parse_optional_u32, parse_optional_usize};

pub(crate) fn run_hf_causal_lm_cuda_decode(args: &mut impl Iterator<Item = String>) -> ExitCode {
    run_hf_causal_lm_cuda_decode_with_reference(args, true)
}

pub(crate) fn run_hf_causal_lm_cuda_device_only_decode(
    args: &mut impl Iterator<Item = String>,
) -> ExitCode {
    run_hf_causal_lm_cuda_decode_with_reference(args, false)
}

fn run_hf_causal_lm_cuda_decode_with_reference(
    args: &mut impl Iterator<Item = String>,
    verify_reference: bool,
) -> ExitCode {
    let path = args.next();
    let input = args.next();
    let steps = match parse_optional_usize(args.next(), 8, "steps") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    exit::print_json_result(hf_causal_lm_cuda_decode_input_json_with_reference(
        path,
        input,
        steps,
        verify_reference,
    ))
}

#[cfg(test)]
pub(crate) fn hf_causal_lm_cuda_decode_json(
    path: Option<String>,
    seed: u32,
    steps: usize,
) -> Result<String, String> {
    let path = path.ok_or_else(|| "hf-cuda-decode requires checkpoint_dir".to_string())?;
    hf_causal_lm_cuda_decode_with_tokens_json(path, "seed_token", vec![seed], None, steps, true)
}

pub(crate) fn hf_causal_lm_cuda_decode_input_json(
    path: Option<String>,
    input: Option<String>,
    steps: usize,
) -> Result<String, String> {
    hf_causal_lm_cuda_decode_input_json_with_reference(path, input, steps, true)
}

pub(crate) fn hf_causal_lm_cuda_device_only_decode_input_json(
    path: Option<String>,
    input: Option<String>,
    steps: usize,
) -> Result<String, String> {
    hf_causal_lm_cuda_decode_input_json_with_reference(path, input, steps, false)
}

fn hf_causal_lm_cuda_decode_input_json_with_reference(
    path: Option<String>,
    input: Option<String>,
    steps: usize,
    verify_reference: bool,
) -> Result<String, String> {
    let path = path.ok_or_else(|| "hf-cuda-decode requires checkpoint_dir".to_string())?;
    let input = input.unwrap_or_else(|| "0".to_string());
    if let Ok(seed) = parse_optional_u32(Some(input.clone()), 0, "seed_token") {
        return hf_causal_lm_cuda_decode_with_tokens_json(
            path,
            "token_id",
            vec![seed],
            None,
            steps,
            verify_reference,
        );
    }
    if let Some(rest) = input.strip_prefix("ids:") {
        return hf_causal_lm_cuda_decode_with_tokens_json(
            path,
            "token_ids",
            parse_token_ids(rest)?,
            None,
            steps,
            verify_reference,
        );
    }
    let tokenizer = Tokenizer::from_file(std::path::Path::new(&path).join("tokenizer.json"))
        .map_err(|err| format!("HF tokenizer load failed: {err}"))?;
    let encoding = tokenizer
        .encode(input.as_str(), false)
        .map_err(|err| format!("HF tokenizer encode failed: {err}"))?;
    hf_causal_lm_cuda_decode_with_tokens_json(
        path,
        "tokenizer_json",
        encoding.get_ids().to_vec(),
        Some(input),
        steps,
        verify_reference,
    )
}

fn hf_causal_lm_cuda_decode_with_tokens_json(
    path: String,
    input_mode: &'static str,
    prompt_token_ids: Vec<u32>,
    prompt_text: Option<String>,
    steps: usize,
    verify_reference: bool,
) -> Result<String, String> {
    let load_options = if verify_reference {
        HfCausalLmLoadOptions::full_verification()
    } else {
        HfCausalLmLoadOptions::skip_payload_hash()
    };
    let loaded = nerva_model::causal_lm::types::HfCausalLmModel::load_from_hf_dir_with_options(
        &path,
        load_options,
    )
    .map_err(|err| format!("HF causal LM load failed: {err:?}"))?;
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let model = &loaded.model;
    let dtype = nerva_model::precision::bits::dtype_label(model.dtype())
        .map_err(|err| format!("HF causal LM dtype failed: {err:?}"))?;
    let prompt_tokens: Vec<TokenId> = prompt_token_ids.iter().copied().map(TokenId).collect();
    let summary = if verify_reference && prompt_tokens.len() == 1 {
        run_loaded_hf_causal_lm_cuda_seed_decode(&runtime, &loaded, prompt_tokens[0], steps, None)
    } else if verify_reference {
        run_loaded_hf_causal_lm_cuda_prompt_decode(&runtime, &loaded, &prompt_tokens, steps, None)
    } else {
        run_loaded_hf_causal_lm_cuda_prompt_decode_device_only(
            &runtime,
            &loaded,
            &prompt_tokens,
            steps,
            None,
        )
    }
    .map_err(|err| format!("HF CUDA causal LM decode failed: {err:?}"))?;
    let generated_text = generated_text_json(&path, &summary.tokens)?;

    Ok(hf_cuda_decode_json(HfCudaDecodeJson {
        path: &path,
        input_mode,
        prompt_text: prompt_text.as_deref(),
        prompt_token_ids: &prompt_token_ids,
        prompt_tokens_len: prompt_tokens.len(),
        seed_token: prompt_tokens.last().map(|token| token.0).unwrap_or(0),
        steps,
        dtype,
        layers: model.layer_count(),
        hidden: model.metadata().hidden_size,
        vocab_size: model.metadata().vocab_size,
        manifest_entries: loaded.summary.manifest.entries.len(),
        shard_plan_entries: loaded.summary.shard_plan.entries.len(),
        tensors_loaded: loaded.summary.tensors_loaded,
        bytes_loaded: loaded.summary.bytes_loaded,
        data_hash: loaded.summary.data_hash,
        data_hash_available: loaded.summary.data_hash_available,
        generated_text: &generated_text,
        summary: &summary,
    }))
}

fn parse_token_ids(value: &str) -> Result<Vec<u32>, String> {
    let mut ids = Vec::new();
    for part in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        ids.push(
            part.parse::<u32>()
                .map_err(|_| "ids prompt must contain unsigned 32-bit integers".to_string())?,
        );
    }
    if ids.is_empty() {
        Err("ids prompt must contain at least one token".to_string())
    } else {
        Ok(ids)
    }
}
