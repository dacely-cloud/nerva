use std::process::ExitCode;

use nerva_core::types::id::token::TokenId;
use nerva_runtime::engine::hf_cuda_decode::run::{
    run_hf_causal_lm_cuda_prompt_decode, run_hf_causal_lm_cuda_seed_decode,
};
use tokenizers::Tokenizer;

use crate::cli::exit;
use crate::cli::model::causal_lm_text::generated_text_json;
use crate::json::json_escape;
use crate::parse::{parse_optional_u32, parse_optional_usize};

pub(crate) fn run_hf_causal_lm_cuda_decode(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let path = args.next();
    let input = args.next();
    let steps = match parse_optional_usize(args.next(), 8, "steps") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    exit::print_json_result(hf_causal_lm_cuda_decode_input_json(path, input, steps))
}

#[cfg(test)]
pub(crate) fn hf_causal_lm_cuda_decode_json(
    path: Option<String>,
    seed: u32,
    steps: usize,
) -> Result<String, String> {
    let path = path.ok_or_else(|| "hf-cuda-decode requires checkpoint_dir".to_string())?;
    hf_causal_lm_cuda_decode_with_tokens_json(path, "seed_token", vec![seed], None, steps)
}

pub(crate) fn hf_causal_lm_cuda_decode_input_json(
    path: Option<String>,
    input: Option<String>,
    steps: usize,
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
        );
    }
    if let Some(rest) = input.strip_prefix("ids:") {
        return hf_causal_lm_cuda_decode_with_tokens_json(
            path,
            "token_ids",
            parse_token_ids(rest)?,
            None,
            steps,
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
    )
}

fn hf_causal_lm_cuda_decode_with_tokens_json(
    path: String,
    input_mode: &'static str,
    prompt_token_ids: Vec<u32>,
    prompt_text: Option<String>,
    steps: usize,
) -> Result<String, String> {
    let loaded = nerva_model::causal_lm::types::HfCausalLmModel::load_from_hf_dir(&path)
        .map_err(|err| format!("HF causal LM load failed: {err:?}"))?;
    let model = loaded.model;
    let dtype = nerva_model::precision::bits::dtype_label(model.dtype())
        .map_err(|err| format!("HF causal LM dtype failed: {err:?}"))?;
    let prompt_tokens: Vec<TokenId> = prompt_token_ids.iter().copied().map(TokenId).collect();
    let summary = if prompt_tokens.len() == 1 {
        run_hf_causal_lm_cuda_seed_decode(&model, prompt_tokens[0], steps)
    } else {
        run_hf_causal_lm_cuda_prompt_decode(&model, &prompt_tokens, steps)
    }
    .map_err(|err| format!("HF CUDA causal LM decode failed: {err:?}"))?;
    let generated_text = generated_text_json(&path, &summary.tokens)?;

    Ok(format!(
        "{{\"status\":\"{}\",\"backend\":\"cuda\",\"path\":\"{}\",\"input_mode\":\"{}\",\"prompt_text\":{},\"prompt_token_ids\":{},\"prompt_tokens\":{},\"seed_token\":{},\"steps\":{},\"dtype\":\"{}\",\"layers\":{},\"hidden\":{},\"vocab_size\":{},\"tokens\":{},\"expected_tokens\":{},\"generated_text\":{},\"parity\":{},\"ledger_count\":{},\"device_events\":{},\"copy_events\":{},\"hard_syncs\":{},\"execution_decisions\":{},\"resident_weight_bytes\":{},\"resident_kv_bytes\":{},\"kv_tokens\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"graph_replays\":{},\"graph_nodes\":{},\"graph_launches\":{},\"graph_replay_events\":{},\"kernel_launches\":{},\"sync_calls\":{},\"host_causality_edges\":{},\"hot_path_allocations\":{},\"output_hash\":{},\"expected_hash\":{},\"error\":{}}}",
        status_json(&summary.status),
        json_escape(&path),
        input_mode,
        json_opt_string(prompt_text.as_deref()),
        u32s_json(&prompt_token_ids),
        prompt_tokens.len(),
        prompt_tokens.last().map(|token| token.0).unwrap_or(0),
        steps,
        dtype,
        model.layer_count(),
        model.metadata().hidden_size,
        model.metadata().vocab_size,
        token_ids_json(&summary.tokens),
        token_ids_json(&summary.expected_tokens),
        generated_text,
        summary.parity,
        summary.ledger_count,
        summary.device_events,
        summary.copy_events,
        summary.hard_syncs,
        summary.execution_decisions,
        summary.resident_weight_bytes,
        summary.resident_kv_bytes,
        summary.kv_tokens,
        summary.h2d_bytes,
        summary.d2h_bytes,
        summary.graph_replays,
        summary.graph_nodes,
        summary.graph_launches,
        summary.graph_replay_events,
        summary.kernel_launches,
        summary.sync_calls,
        summary.host_causality_edges,
        summary.hot_path_allocations,
        summary.output_hash,
        summary.expected_hash,
        json_opt_string(summary.error.as_deref()),
    ))
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

fn status_json(status: &nerva_cuda::smoke::status::SmokeStatus) -> &'static str {
    match status {
        nerva_cuda::smoke::status::SmokeStatus::Ok => "ok",
        nerva_cuda::smoke::status::SmokeStatus::Unavailable => "unavailable",
        nerva_cuda::smoke::status::SmokeStatus::Failed => "failed",
    }
}

fn token_ids_json(tokens: &[TokenId]) -> String {
    let mut out = String::from("[");
    for (index, token) in tokens.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&token.0.to_string());
    }
    out.push(']');
    out
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

fn json_opt_string(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("\"{}\"", json_escape(value)),
        None => "null".to_string(),
    }
}
