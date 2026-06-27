use std::process::ExitCode;

use nerva_core::types::id::token::TokenId;
use nerva_model::weights::layout::entry::WeightBlockRole;
use tokenizers::Tokenizer;

use crate::cli::exit;
use crate::cli::model::causal_lm_text::generated_text_json;
use crate::json::json_escape;
use crate::parse::{parse_optional_u32, parse_optional_usize};

pub(crate) fn run_hf_causal_lm_decode(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let path = args.next();
    let input = args.next();
    let steps = match parse_optional_usize(args.next(), 8, "steps") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    exit::print_json_result(hf_causal_lm_decode_input_json(path, input, steps))
}

pub(crate) fn hf_causal_lm_decode_input_json(
    path: Option<String>,
    input: Option<String>,
    steps: usize,
) -> Result<String, String> {
    let path = path.ok_or_else(|| "hf-decode requires checkpoint_dir".to_string())?;
    let input = input.unwrap_or_else(|| "0".to_string());
    if let Ok(seed) = parse_optional_u32(Some(input.clone()), 0, "seed_token") {
        return hf_causal_lm_decode_with_tokens_json(path, "token_id", vec![seed], None, steps);
    }
    if let Some(rest) = input.strip_prefix("ids:") {
        return hf_causal_lm_decode_with_tokens_json(
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
    hf_causal_lm_decode_with_tokens_json(
        path,
        "tokenizer_json",
        encoding.get_ids().to_vec(),
        Some(input),
        steps,
    )
}

fn hf_causal_lm_decode_with_tokens_json(
    path: String,
    input_mode: &'static str,
    prompt_token_ids: Vec<u32>,
    prompt_text: Option<String>,
    steps: usize,
) -> Result<String, String> {
    let loaded = nerva_model::causal_lm::types::HfCausalLmModel::load_from_hf_dir(&path)
        .map_err(|err| format!("HF causal LM load failed: {err:?}"))?;
    let summary = loaded.summary;
    let model = loaded.model;
    let dtype = nerva_model::precision::bits::dtype_label(model.dtype())
        .map_err(|err| format!("HF causal LM dtype failed: {err:?}"))?;
    let max_context = prompt_token_ids
        .len()
        .checked_add(steps)
        .ok_or_else(|| "HF causal LM context length overflow".to_string())?;
    let mut scratch = nerva_model::causal_lm::types::HfCausalLmDecodeScratch::new_with_context(
        model.shape(),
        model.metadata().vocab_size,
        model.layer_count(),
        max_context,
    )
    .map_err(|err| format!("HF causal LM scratch failed: {err:?}"))?;
    let prompt_tokens: Vec<TokenId> = prompt_token_ids.iter().copied().map(TokenId).collect();
    let output = model
        .decode_greedy_from_prompt_tokens(&prompt_tokens, steps, &mut scratch)
        .map_err(|err| format!("HF causal LM decode failed: {err:?}"))?;
    let final_norm_manifest = summary
        .manifest
        .entries
        .iter()
        .any(|entry| entry.role == WeightBlockRole::FinalNorm);
    let ledger_events: usize = output
        .ledgers
        .iter()
        .map(|ledger| ledger.events.len())
        .sum();
    let execution_decisions: u64 = output
        .ledgers
        .iter()
        .map(|ledger| ledger.execution_decisions.len() as u64)
        .sum();
    let hot_path_allocations: u64 = output
        .ledgers
        .iter()
        .map(|ledger| ledger.hot_path_allocations)
        .sum();
    let generated_text = generated_text_json(&path, &output.generated_tokens)?;

    Ok(format!(
        "{{\"status\":\"ok\",\"path\":\"{}\",\"input_mode\":\"{}\",\"context_mode\":\"{}\",\"stop_reason\":\"{}\",\"prompt_text\":{},\"prompt_token_ids\":{},\"prompt_tokens\":{},\"dtype\":\"{}\",\"layers\":{},\"hidden\":{},\"vocab_size\":{},\"seed_token\":{},\"steps\":{},\"tokens\":{},\"generated_text\":{},\"output_hash\":{},\"manifest_entries\":{},\"shard_plan_entries\":{},\"tensors_loaded\":{},\"bytes_loaded\":{},\"data_hash\":{},\"data_hash_available\":{},\"final_norm_manifest\":{},\"tied_lm_head\":{},\"ledger_count\":{},\"ledger_events\":{},\"execution_decisions\":{},\"hot_path_allocations\":{}}}",
        json_escape(&path),
        input_mode,
        output.context_mode.as_str(),
        output.stop_reason.as_str(),
        json_opt_string(prompt_text.as_deref()),
        u32s_json(&prompt_token_ids),
        output.prompt_tokens.len(),
        dtype,
        model.layer_count(),
        model.metadata().hidden_size,
        model.metadata().vocab_size,
        output.seed_token.0,
        steps,
        token_ids_json(&output.generated_tokens),
        generated_text,
        token_hash(&output.generated_tokens),
        summary.manifest.entries.len(),
        summary.shard_plan.entries.len(),
        summary.tensors_loaded,
        summary.bytes_loaded,
        summary.data_hash,
        summary.data_hash_available,
        final_norm_manifest,
        summary.tied_lm_head,
        output.ledgers.len(),
        ledger_events,
        execution_decisions,
        hot_path_allocations,
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

fn json_opt_string(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("\"{}\"", json_escape(value)),
        None => "null".to_string(),
    }
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

fn token_hash(tokens: &[TokenId]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for token in tokens {
        for byte in token.0.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}
