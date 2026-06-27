use crate::decode::hf_sequence::footprint::CudaHfDecodeSequenceFootprint;
use crate::json::{json_opt_bool, json_opt_str, json_opt_usize};
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
    pub planned_footprint: CudaHfDecodeSequenceFootprint,
    pub device_total_memory_bytes: Option<usize>,
    pub device_free_memory_bytes: Option<usize>,
    pub fits_device_free_memory: Option<bool>,
    pub resident_weight_bytes: u64,
    pub planned_weight_blocks: u32,
    pub planned_gpu_resident_blocks: u32,
    pub planned_gpu_staged_blocks: u32,
    pub planned_weight_bytes: u64,
    pub planned_gpu_resident_weight_bytes: u64,
    pub planned_gpu_staged_weight_bytes: u64,
    pub descriptor_gpu_resident_h2d_bytes: u64,
    pub descriptor_gpu_staged_h2d_bytes: u64,
    pub planned_weight_descriptor_count: u32,
    pub planned_weight_descriptor_hash: u64,
    pub resident_kv_bytes: u64,
    pub kv_tokens: u64,
    pub device_arena_bytes: u64,
    pub pinned_host_bytes: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub graph_replays: u64,
    pub graph_nodes: u64,
    pub graph_launches: u64,
    pub graph_captures: u64,
    pub graph_cache_hits: u64,
    pub kernel_launches: u64,
    pub device_elapsed_ns: u64,
    pub projection_ns: u64,
    pub qkv_projection_ns: u64,
    pub attention_output_projection_ns: u64,
    pub gate_up_projection_ns: u64,
    pub down_projection_ns: u64,
    pub lm_head_projection_ns: u64,
    pub attention_ns: u64,
    pub mlp_ns: u64,
    pub norm_ns: u64,
    pub sampling_ns: u64,
    pub sync_calls: u64,
    pub host_causality_edges: u64,
    pub hot_path_allocations: u64,
    pub error: Option<String>,
}

impl CudaHfDecodeSequenceSummary {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"status\":\"{}\",\"dtype\":{},\"hidden\":{},\"heads\":{},\"kv_heads\":{},\"head_dim\":{},\"intermediate\":{},\"vocab_size\":{},\"layer_count\":{},\"steps\":{},\"seed_token\":{},\"tokens\":{},\"observed_tokens\":{},\"observed_token_hash\":{},\"planned_footprint\":{},\"device_total_memory_bytes\":{},\"device_free_memory_bytes\":{},\"fits_device_free_memory\":{},\"resident_weight_bytes\":{},\"planned_weight_blocks\":{},\"planned_gpu_resident_blocks\":{},\"planned_gpu_staged_blocks\":{},\"planned_weight_bytes\":{},\"planned_gpu_resident_weight_bytes\":{},\"planned_gpu_staged_weight_bytes\":{},\"descriptor_gpu_resident_H2D_bytes\":{},\"descriptor_gpu_staged_H2D_bytes\":{},\"planned_weight_descriptor_count\":{},\"planned_weight_descriptor_hash\":{},\"resident_kv_bytes\":{},\"kv_tokens\":{},\"device_arena_bytes\":{},\"pinned_host_bytes\":{},\"H2D_bytes\":{},\"D2H_bytes\":{},\"graph_replays\":{},\"graph_nodes\":{},\"graph_launches\":{},\"graph_captures\":{},\"graph_cache_hits\":{},\"kernel_launches\":{},\"device_elapsed_ns\":{},\"projection_ns\":{},\"qkv_projection_ns\":{},\"attention_output_projection_ns\":{},\"gate_up_projection_ns\":{},\"down_projection_ns\":{},\"lm_head_projection_ns\":{},\"attention_ns\":{},\"mlp_ns\":{},\"norm_ns\":{},\"sampling_ns\":{},\"sync_calls\":{},\"host_causality_edges\":{},\"hot_path_allocations\":{},\"error\":{}}}",
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
            self.planned_footprint.to_json(),
            json_opt_usize(self.device_total_memory_bytes),
            json_opt_usize(self.device_free_memory_bytes),
            json_opt_bool(self.fits_device_free_memory),
            self.resident_weight_bytes,
            self.planned_weight_blocks,
            self.planned_gpu_resident_blocks,
            self.planned_gpu_staged_blocks,
            self.planned_weight_bytes,
            self.planned_gpu_resident_weight_bytes,
            self.planned_gpu_staged_weight_bytes,
            self.descriptor_gpu_resident_h2d_bytes,
            self.descriptor_gpu_staged_h2d_bytes,
            self.planned_weight_descriptor_count,
            self.planned_weight_descriptor_hash,
            self.resident_kv_bytes,
            self.kv_tokens,
            self.device_arena_bytes,
            self.pinned_host_bytes,
            self.h2d_bytes,
            self.d2h_bytes,
            self.graph_replays,
            self.graph_nodes,
            self.graph_launches,
            self.graph_captures,
            self.graph_cache_hits,
            self.kernel_launches,
            self.device_elapsed_ns,
            self.projection_ns,
            self.qkv_projection_ns,
            self.attention_output_projection_ns,
            self.gate_up_projection_ns,
            self.down_projection_ns,
            self.lm_head_projection_ns,
            self.attention_ns,
            self.mlp_ns,
            self.norm_ns,
            self.sampling_ns,
            self.sync_calls,
            self.host_causality_edges,
            self.hot_path_allocations,
            json_opt_str(self.error.as_deref()),
        )
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
