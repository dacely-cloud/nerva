use std::process::ExitCode;

use nerva_core::types::id::token::TokenId;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_runtime::engine::hf_cuda_decode::file_backed::session::create_hf_causal_lm_cuda_shard_backed_device_only_session;
use nerva_runtime::engine::runtime::{Runtime, RuntimeConfig};

use crate::cli::exit;
use crate::cli::model::causal_lm_text::generated_text_json;
use crate::json::json_escape;
use crate::parse::parse_optional_usize;

pub(crate) fn run_hf_causal_lm_cuda_device_session_decode(
    args: &mut impl Iterator<Item = String>,
) -> ExitCode {
    let path = args.next();
    let max_context = match parse_optional_usize(args.next(), 8, "max_context_tokens") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let steps = match parse_optional_usize(args.next(), 1, "steps") {
        Ok(value) => value,
        Err(reason) => return exit::parse_error(reason),
    };
    let prompts = args.collect::<Vec<_>>();
    exit::print_json_result(hf_causal_lm_cuda_device_session_json(
        path,
        max_context,
        steps,
        prompts,
    ))
}

pub(crate) fn hf_causal_lm_cuda_device_session_json(
    path: Option<String>,
    max_context_tokens: usize,
    steps: usize,
    prompt_specs: Vec<String>,
) -> Result<String, String> {
    let path =
        path.ok_or_else(|| "hf-cuda-decode-device-session requires checkpoint_dir".to_string())?;
    let runtime = Runtime::new(RuntimeConfig::default())
        .map_err(|err| format!("runtime init failed: {err:?}"))?;
    let mut session = create_hf_causal_lm_cuda_shard_backed_device_only_session(
        &runtime,
        &path,
        max_context_tokens,
        None,
    )
    .map_err(|err| format!("HF CUDA session create failed: {err:?}"))?;
    let prompts = if prompt_specs.is_empty() {
        vec!["0".to_string()]
    } else {
        prompt_specs
    };
    let mut runs = String::from("[");
    let mut ok = true;
    for (index, prompt) in prompts.iter().enumerate() {
        if index > 0 {
            runs.push(',');
        }
        let prompt_ids = parse_prompt(prompt)?;
        let token_ids = prompt_ids.iter().copied().map(TokenId).collect::<Vec<_>>();
        let summary = session
            .run(&token_ids, steps)
            .map_err(|err| format!("HF CUDA session run failed: {err:?}"))?;
        ok &= summary.status == SmokeStatus::Ok;
        let generated_text = generated_text_json(&path, &summary.tokens)?;
        runs.push_str(&format!(
            "{{\"input\":\"{}\",\"prompt_token_ids\":{},\"prompt_tokens\":{},\"seed_token\":{},\"generated_text\":{},\"summary\":{}}}",
            json_escape(prompt),
            u32s_json(&prompt_ids),
            prompt_ids.len(),
            prompt_ids.last().copied().unwrap_or(0),
            generated_text,
            summary.to_json(),
        ));
    }
    runs.push(']');
    let dtype = nerva_model::precision::bits::dtype_label(session.dtype)
        .map_err(|err| format!("HF causal LM dtype failed: {err:?}"))?;
    Ok(format!(
        "{{\"status\":\"{}\",\"backend\":\"cuda\",\"mode\":\"device_session\",\"path\":\"{}\",\"max_context_tokens\":{},\"steps\":{},\"dtype\":\"{}\",\"layers\":{},\"hidden\":{},\"vocab_size\":{},\"manifest_entries\":{},\"shard_plan_entries\":{},\"tensors_loaded\":{},\"bytes_loaded\":{},\"data_hash\":{},\"data_hash_available\":{},\"create\":{},\"runs\":{}}}",
        if ok { "ok" } else { "failed" },
        json_escape(&path),
        max_context_tokens,
        steps,
        dtype,
        session.metadata.num_hidden_layers,
        session.metadata.hidden_size,
        session.metadata.vocab_size,
        session.manifest_entries,
        session.shard_plan_entries,
        session.tensors_loaded,
        session.bytes_loaded,
        session.data_hash,
        session.data_hash_available,
        session.create_summary.to_json(),
        runs,
    ))
}

fn parse_prompt(value: &str) -> Result<Vec<u32>, String> {
    if let Some(rest) = value.strip_prefix("ids:") {
        return parse_token_ids(rest);
    }
    value
        .parse::<u32>()
        .map(|token| vec![token])
        .map_err(|_| "session prompt must be a token id or ids:a,b,c".to_string())
}

fn parse_token_ids(value: &str) -> Result<Vec<u32>, String> {
    let ids = value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u32>()
                .map_err(|_| "ids prompt must contain unsigned 32-bit integers".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ids.is_empty() {
        Err("ids prompt must contain at least one token".to_string())
    } else {
        Ok(ids)
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
