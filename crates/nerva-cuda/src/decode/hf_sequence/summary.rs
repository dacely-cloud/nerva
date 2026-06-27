use crate::json::json_opt_str;
use crate::smoke::status::SmokeStatus;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaHfDecodeSequenceSummary {
    pub status: SmokeStatus,
    pub dtype: u32,
    pub hidden: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub intermediate: u32,
    pub vocab_size: u32,
    pub layer_count: u32,
    pub steps: u32,
    pub seed_token: u32,
    pub tokens: Vec<u32>,
    pub observed_token_hash: u64,
    pub resident_weight_bytes: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub kernel_launches: u64,
    pub sync_calls: u64,
    pub host_causality_edges: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaHfDecodeSequenceSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"status\":\"{}\",\"dtype\":{},\"hidden\":{},\"heads\":{},\"kv_heads\":{},\"head_dim\":{},\"intermediate\":{},\"vocab_size\":{},\"layer_count\":{},\"steps\":{},\"seed_token\":{},\"tokens\":{},\"observed_tokens\":{},\"observed_token_hash\":{},\"resident_weight_bytes\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"kernel_launches\":{},\"sync_calls\":{},\"host_causality_edges\":{},\"hot_path_allocations\":{},\"error\":{}}}",
            status_str(&self.status),
            self.dtype,
            self.hidden,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.intermediate,
            self.vocab_size,
            self.layer_count,
            self.steps,
            self.seed_token,
            u32s_json(&self.tokens),
            self.tokens.len(),
            self.observed_token_hash,
            self.resident_weight_bytes,
            self.device_arena_bytes,
            self.pinned_host_bytes,
            self.h2d_bytes,
            self.d2h_bytes,
            self.kernel_launches,
            self.sync_calls,
            self.host_causality_edges,
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
    }
}

pub(crate) fn empty_summary(
    status: SmokeStatus,
    dtype: u32,
    hidden: usize,
    vocab_size: usize,
    steps: usize,
    seed_token: u32,
    error: String,
) -> CudaHfDecodeSequenceSummary {
    CudaHfDecodeSequenceSummary {
        status,
        dtype,
        hidden: hidden as u32,
        heads: 0,
        kv_heads: 0,
        head_dim: 0,
        intermediate: 0,
        vocab_size: vocab_size as u32,
        layer_count: 0,
        steps: steps as u32,
        seed_token,
        tokens: Vec::new(),
        observed_token_hash: 0,
        resident_weight_bytes: 0,
        device_arena_bytes: 0,
        pinned_host_bytes: 0,
        h2d_bytes: 0,
        d2h_bytes: 0,
        kernel_launches: 0,
        sync_calls: 0,
        host_causality_edges: 0,
        hot_path_allocations: 0,
        error: Some(error),
    }
}

fn status_str(status: &SmokeStatus) -> &'static str {
    match status {
        SmokeStatus::Ok => "ok",
        SmokeStatus::Unavailable => "unavailable",
        SmokeStatus::Failed => "failed",
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
