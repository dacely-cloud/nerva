use std::process::ExitCode;

use nerva_core::types::id::token::TokenId;
use nerva_runtime::engine::hf_cuda_decode::run::run_hf_causal_lm_cuda_seed_decode;

use crate::cli::exit;
use crate::json::json_escape;
use crate::parse::{parse_optional_u32, parse_optional_usize};

pub(crate) fn run_hf_causal_lm_cuda_decode(args: &mut impl Iterator<Item = String>) -> ExitCode {
    let path = args.next();
    let seed = match parse_optional_u32(args.next(), 0, "seed_token") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let steps = match parse_optional_usize(args.next(), 8, "steps") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    exit::print_json_result(hf_causal_lm_cuda_decode_json(path, seed, steps))
}

pub(crate) fn hf_causal_lm_cuda_decode_json(
    path: Option<String>,
    seed: u32,
    steps: usize,
) -> Result<String, String> {
    let path = path.ok_or_else(|| "hf-cuda-decode requires checkpoint_dir".to_string())?;
    let loaded = nerva_model::causal_lm::types::HfCausalLmModel::load_from_hf_dir(&path)
        .map_err(|err| format!("HF causal LM load failed: {err:?}"))?;
    let model = loaded.model;
    let dtype = nerva_model::precision::bits::dtype_label(model.dtype())
        .map_err(|err| format!("HF causal LM dtype failed: {err:?}"))?;
    let summary = run_hf_causal_lm_cuda_seed_decode(&model, TokenId(seed), steps)
        .map_err(|err| format!("HF CUDA causal LM decode failed: {err:?}"))?;

    Ok(format!(
        "{{\"status\":\"{}\",\"backend\":\"cuda\",\"path\":\"{}\",\"input_mode\":\"seed_token\",\"seed_token\":{},\"steps\":{},\"dtype\":\"{}\",\"layers\":{},\"hidden\":{},\"vocab_size\":{},\"tokens\":{},\"expected_tokens\":{},\"parity\":{},\"ledger_count\":{},\"device_events\":{},\"copy_events\":{},\"hard_syncs\":{},\"execution_decisions\":{},\"resident_weight_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"host_causality_edges\":{},\"hot_path_allocations\":{},\"output_hash\":{},\"expected_hash\":{},\"error\":{}}}",
        status_json(&summary.status),
        json_escape(&path),
        seed,
        steps,
        dtype,
        model.layer_count(),
        model.metadata().hidden_size,
        model.metadata().vocab_size,
        token_ids_json(&summary.tokens),
        token_ids_json(&summary.expected_tokens),
        summary.parity,
        summary.ledger_count,
        summary.device_events,
        summary.copy_events,
        summary.hard_syncs,
        summary.execution_decisions,
        summary.resident_weight_bytes,
        summary.h2d_bytes,
        summary.d2h_bytes,
        summary.kernel_launches,
        summary.sync_calls,
        summary.host_causality_edges,
        summary.hot_path_allocations,
        summary.output_hash,
        summary.expected_hash,
        json_opt_string(summary.error.as_deref()),
    ))
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

fn json_opt_string(value: Option<&str>) -> String {
    match value {
        Some(value) => format!("\"{}\"", json_escape(value)),
        None => "null".to_string(),
    }
}
