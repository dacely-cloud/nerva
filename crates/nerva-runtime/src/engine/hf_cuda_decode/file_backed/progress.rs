use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;

use crate::engine::hf_cuda_decode::summary::HfCudaSeedDecodeSummary;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum HfCudaDeviceProgressPhase {
    Prefill,
    #[default]
    Decode,
}

impl HfCudaDeviceProgressPhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct HfCudaDeviceSessionChunkProgress {
    pub phase: HfCudaDeviceProgressPhase,
    pub generated: usize,
    pub requested: usize,
    pub chunk_requested: usize,
    pub chunk_index: usize,
    pub observed: usize,
    pub hit_stop: bool,
    pub wall_ns: u64,
    pub device_ns: u64,
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
    pub graph_nodes: u64,
    pub graph_replays: u64,
    pub graph_cache_hits: u64,
    pub kernel_launches: u64,
    pub kv_tokens: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub sync_calls: u64,
    pub host_causality_edges: u64,
    pub hot_path_allocations: u64,
    pub tokens: Vec<TokenId>,
}

impl HfCudaDeviceSessionChunkProgress {
    pub fn prefill_started(requested: usize, prompt_tokens: usize) -> Self {
        Self {
            phase: HfCudaDeviceProgressPhase::Prefill,
            generated: 0,
            requested,
            chunk_requested: requested,
            chunk_index: 0,
            observed: prompt_tokens,
            hit_stop: false,
            wall_ns: 0,
            device_ns: 0,
            projection_ns: 0,
            qkv_projection_ns: 0,
            attention_output_projection_ns: 0,
            gate_up_projection_ns: 0,
            down_projection_ns: 0,
            lm_head_projection_ns: 0,
            attention_ns: 0,
            mlp_ns: 0,
            norm_ns: 0,
            sampling_ns: 0,
            graph_nodes: 0,
            graph_replays: 0,
            graph_cache_hits: 0,
            kernel_launches: 0,
            kv_tokens: 0,
            h2d_bytes: 0,
            d2h_bytes: 0,
            sync_calls: 0,
            host_causality_edges: 0,
            hot_path_allocations: 0,
            tokens: Vec::new(),
        }
    }

    pub fn from_summary(
        generated: usize,
        requested: usize,
        chunk_index: usize,
        hit_stop: bool,
        summary: &HfCudaSeedDecodeSummary,
    ) -> Self {
        Self {
            phase: HfCudaDeviceProgressPhase::Decode,
            generated,
            requested,
            chunk_requested: summary.steps_requested,
            chunk_index,
            observed: summary.tokens.len(),
            hit_stop,
            wall_ns: summary
                .critical_paths
                .iter()
                .map(|path| path.wall_latency_ns)
                .sum(),
            device_ns: summary
                .critical_paths
                .iter()
                .map(|path| path.device_timeline_active_ns)
                .sum(),
            projection_ns: summary.projection_ns,
            qkv_projection_ns: summary.qkv_projection_ns,
            attention_output_projection_ns: summary.attention_output_projection_ns,
            gate_up_projection_ns: summary.gate_up_projection_ns,
            down_projection_ns: summary.down_projection_ns,
            lm_head_projection_ns: summary.lm_head_projection_ns,
            attention_ns: summary.attention_ns,
            mlp_ns: summary.mlp_ns,
            norm_ns: summary.norm_ns,
            sampling_ns: summary.sampling_ns,
            graph_nodes: summary.graph_nodes,
            graph_replays: summary.graph_replays,
            graph_cache_hits: summary.graph_cache_hits,
            kernel_launches: summary.kernel_launches,
            kv_tokens: summary.kv_tokens,
            h2d_bytes: summary.h2d_bytes,
            d2h_bytes: summary.d2h_bytes,
            sync_calls: summary.sync_calls,
            host_causality_edges: summary.host_causality_edges,
            hot_path_allocations: summary.hot_path_allocations,
            tokens: summary.tokens.clone(),
        }
    }

    pub fn from_prefill_summary(
        requested: usize,
        wall_ns: u64,
        summary: &CudaHfDecodeSequenceSummary,
    ) -> Self {
        Self {
            phase: HfCudaDeviceProgressPhase::Prefill,
            generated: 0,
            requested,
            chunk_requested: requested,
            chunk_index: 0,
            observed: summary.kv_tokens as usize,
            hit_stop: false,
            wall_ns,
            device_ns: summary.device_elapsed_ns,
            projection_ns: summary.projection_ns,
            qkv_projection_ns: summary.qkv_projection_ns,
            attention_output_projection_ns: summary.attention_output_projection_ns,
            gate_up_projection_ns: summary.gate_up_projection_ns,
            down_projection_ns: summary.down_projection_ns,
            lm_head_projection_ns: summary.lm_head_projection_ns,
            attention_ns: summary.attention_ns,
            mlp_ns: summary.mlp_ns,
            norm_ns: summary.norm_ns,
            sampling_ns: summary.sampling_ns,
            graph_nodes: summary.graph_nodes,
            graph_replays: summary.graph_replays,
            graph_cache_hits: summary.graph_cache_hits,
            kernel_launches: summary.kernel_launches,
            kv_tokens: summary.kv_tokens,
            h2d_bytes: summary.h2d_bytes,
            d2h_bytes: summary.d2h_bytes,
            sync_calls: summary.sync_calls,
            host_causality_edges: summary.host_causality_edges,
            hot_path_allocations: summary.hot_path_allocations,
            tokens: summary.tokens.iter().copied().map(TokenId).collect(),
        }
    }
}
