use nerva_core::types::id::token::TokenId;
use nerva_runtime::engine::hf_cuda_decode::summary::HfCudaSeedDecodeSummary;

use crate::json::json_escape;

pub(crate) struct HfCudaDecodeJson<'a> {
    pub path: &'a str,
    pub input_mode: &'static str,
    pub prompt_text: Option<&'a str>,
    pub prompt_token_ids: &'a [u32],
    pub prompt_tokens_len: usize,
    pub seed_token: u32,
    pub steps: usize,
    pub dtype: &'a str,
    pub layers: usize,
    pub hidden: usize,
    pub vocab_size: usize,
    pub generated_text: &'a str,
    pub summary: &'a HfCudaSeedDecodeSummary,
}

pub(crate) fn hf_cuda_decode_json(input: HfCudaDecodeJson<'_>) -> String {
    let summary = input.summary;
    format!(
        "{{\"status\":\"{}\",\"backend\":\"cuda\",\"path\":\"{}\",\"input_mode\":\"{}\",\"prompt_text\":{},\"prompt_token_ids\":{},\"prompt_tokens\":{},\"seed_token\":{},\"steps\":{},\"dtype\":\"{}\",\"layers\":{},\"hidden\":{},\"vocab_size\":{},\"tokens\":{},\"expected_tokens\":{},\"generated_text\":{},\"parity\":{},\"ledger_count\":{},\"device_events\":{},\"copy_events\":{},\"hard_syncs\":{},\"soft_visibility_syncs\":{},\"execution_decisions\":{},\"resident_weight_bytes\":{},\"resident_kv_bytes\":{},\"kv_tokens\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"graph_replays\":{},\"graph_nodes\":{},\"graph_launches\":{},\"graph_replay_events\":{},\"kernel_launches\":{},\"sync_calls\":{},\"host_causality_edges\":{},\"hot_path_allocations\":{},\"output_hash\":{},\"expected_hash\":{},\"resident_weight_plan\":{},\"critical_paths\":{},\"error\":{}}}",
        status_json(&summary.status),
        json_escape(input.path),
        input.input_mode,
        json_opt_string(input.prompt_text),
        u32s_json(input.prompt_token_ids),
        input.prompt_tokens_len,
        input.seed_token,
        input.steps,
        input.dtype,
        input.layers,
        input.hidden,
        input.vocab_size,
        token_ids_json(&summary.tokens),
        token_ids_json(&summary.expected_tokens),
        input.generated_text,
        summary.parity,
        summary.ledger_count,
        summary.device_events,
        summary.copy_events,
        summary.hard_syncs,
        summary.soft_visibility_syncs,
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
        summary.resident_weights.to_json(),
        summary.critical_paths_json(),
        json_opt_string(summary.error.as_deref()),
    )
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
