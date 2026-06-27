use std::process::ExitCode;

use nerva_core::types::id::token::TokenId;
use nerva_model::weights::layout::entry::WeightBlockRole;

use crate::cli::exit;
use crate::json::json_escape;
use crate::parse::{parse_optional_u32, parse_optional_usize};

pub(crate) fn run_hf_causal_lm_decode(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let path = args.next();
    let seed_token = match parse_optional_u32(args.next(), 0, "seed_token") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let steps = match parse_optional_usize(args.next(), 8, "steps") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    exit::print_json_result(hf_causal_lm_decode_json(path, seed_token, steps))
}

pub(crate) fn hf_causal_lm_decode_json(
    path: Option<String>,
    seed_token: u32,
    steps: usize,
) -> Result<String, String> {
    let path = path.ok_or_else(|| "hf-decode requires checkpoint_dir".to_string())?;
    let loaded = nerva_model::causal_lm::types::HfCausalLmModel::load_from_hf_dir(&path)
        .map_err(|err| format!("HF causal LM load failed: {err:?}"))?;
    let summary = loaded.summary;
    let model = loaded.model;
    let dtype = nerva_model::precision::bits::dtype_label(model.dtype())
        .map_err(|err| format!("HF causal LM dtype failed: {err:?}"))?;
    let mut scratch = nerva_model::causal_lm::types::HfCausalLmDecodeScratch::new(
        model.shape(),
        model.metadata().vocab_size,
    )
    .map_err(|err| format!("HF causal LM scratch failed: {err:?}"))?;
    let (tokens, ledgers) = model
        .decode_greedy(TokenId(seed_token), steps, &mut scratch)
        .map_err(|err| format!("HF causal LM decode failed: {err:?}"))?;
    let final_norm_manifest = summary
        .manifest
        .entries
        .iter()
        .any(|entry| entry.role == WeightBlockRole::FinalNorm);
    let ledger_events: usize = ledgers.iter().map(|ledger| ledger.events.len()).sum();
    let execution_decisions: u64 = ledgers
        .iter()
        .map(|ledger| ledger.execution_decisions.len() as u64)
        .sum();
    let hot_path_allocations: u64 = ledgers
        .iter()
        .map(|ledger| ledger.hot_path_allocations)
        .sum();

    Ok(format!(
        "{{\"status\":\"ok\",\"path\":\"{}\",\"dtype\":\"{}\",\"layers\":{},\"hidden\":{},\"vocab_size\":{},\"seed_token\":{},\"steps\":{},\"tokens\":{},\"output_hash\":{},\"manifest_entries\":{},\"shard_plan_entries\":{},\"tensors_loaded\":{},\"bytes_loaded\":{},\"data_hash\":{},\"final_norm_manifest\":{},\"tied_lm_head\":{},\"ledger_count\":{},\"ledger_events\":{},\"execution_decisions\":{},\"hot_path_allocations\":{}}}",
        json_escape(&path),
        dtype,
        model.layer_count(),
        model.metadata().hidden_size,
        model.metadata().vocab_size,
        seed_token,
        steps,
        token_ids_json(&tokens),
        token_hash(&tokens),
        summary.manifest.entries.len(),
        summary.shard_plan.entries.len(),
        summary.tensors_loaded,
        summary.bytes_loaded,
        summary.data_hash,
        final_norm_manifest,
        summary.tied_lm_head,
        ledgers.len(),
        ledger_events,
        execution_decisions,
        hot_path_allocations,
    ))
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
