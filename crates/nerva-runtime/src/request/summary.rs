use nerva_core::types::id::token::TokenId;

use crate::request::types::StopReason;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RequestStateProbeStatus {
    Ok,
}

fn json_token_array(tokens: &[TokenId]) -> String {
    let mut out = String::from("[");
    for (index, token) in tokens.iter().enumerate() {
        if index != 0 {
            out.push(',');
        }
        out.push_str(&token.0.to_string());
    }
    out.push(']');
    out
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestStateSummary {
    pub status: RequestStateProbeStatus,
    pub prompt_tokens: Vec<TokenId>,
    pub generated_tokens: Vec<TokenId>,
    pub host_observed_tokens: Vec<TokenId>,
    pub seed_from_prompt: bool,
    pub device_generated_edges: u64,
    pub device_steps_without_host_observation: u64,
    pub max_host_visibility_lag: usize,
    pub stop_reason: StopReason,
    pub duplicate_row_rejections: u64,
    pub missing_row_rejections: u64,
    pub post_completion_rejections: u64,
    pub ledger_count: u64,
    pub device_events: u64,
    pub hot_path_allocations: u64,
}

impl RequestStateSummary {
    pub fn passed(&self) -> bool {
        self.seed_from_prompt
            && self.device_generated_edges > 0
            && self.device_steps_without_host_observation > 0
            && self.generated_tokens == self.host_observed_tokens
            && self.stop_reason == StopReason::EosToken
            && self.duplicate_row_rejections == 1
            && self.missing_row_rejections == 1
            && self.post_completion_rejections == 1
            && self.hot_path_allocations == 0
    }

    pub fn to_json(&self) -> String {
        let status = match self.status {
            RequestStateProbeStatus::Ok => "ok",
        };
        format!(
            "{{\"status\":\"{}\",\"prompt_tokens\":{},\"generated_tokens\":{},\"host_observed_tokens\":{},\"seed_from_prompt\":{},\"device_generated_edges\":{},\"device_steps_without_host_observation\":{},\"max_host_visibility_lag\":{},\"stop_reason\":\"{}\",\"duplicate_row_rejections\":{},\"missing_row_rejections\":{},\"post_completion_rejections\":{},\"ledger_count\":{},\"device_events\":{},\"hot_path_allocations\":{}}}",
            status,
            json_token_array(&self.prompt_tokens),
            json_token_array(&self.generated_tokens),
            json_token_array(&self.host_observed_tokens),
            self.seed_from_prompt,
            self.device_generated_edges,
            self.device_steps_without_host_observation,
            self.max_host_visibility_lag,
            self.stop_reason.as_str(),
            self.duplicate_row_rejections,
            self.missing_row_rejections,
            self.post_completion_rejections,
            self.ledger_count,
            self.device_events,
            self.hot_path_allocations,
        )
    }
}
