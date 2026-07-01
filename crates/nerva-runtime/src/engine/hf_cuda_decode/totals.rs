use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_sequence::footprint::CudaHfDecodeSequenceFootprint;
use nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use nerva_cuda::smoke::status::SmokeStatus;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::sync::SyncClass;
use nerva_ledger::types::token::ledger::TokenLedger;

use crate::engine::hf_cuda_decode::hash::hash_tokens;
use crate::engine::hf_cuda_decode::summary::{
    HfCudaResidentWeightSummary, HfCudaSeedDecodeSummary,
};
use crate::engine::hf_cuda_decode::totals_counts::{
    critical_paths, event_count, execution_decisions, hot_path_allocations, sync_count,
};

#[derive(Default)]
pub(super) struct CudaDecodeCounters {
    resident_weight_bytes: u64,
    resident_kv_bytes: u64,
    kv_tokens: u64,
    h2d_bytes: u64,
    d2h_bytes: u64,
    graph_replays: u64,
    graph_nodes: u64,
    graph_launches: u64,
    graph_captures: u64,
    graph_cache_hits: u64,
    kernel_launches: u64,
    experimental_rt_selector_launches: u64,
    experimental_rt_sparse_attention_chunks: u32,
    experimental_rt_dense_attention_chunks: u32,
    experimental_rt_attention_chunks: u32,
    projection_ns: u64,
    qkv_projection_ns: u64,
    attention_output_projection_ns: u64,
    gate_up_projection_ns: u64,
    down_projection_ns: u64,
    lm_head_projection_ns: u64,
    attention_ns: u64,
    mlp_ns: u64,
    norm_ns: u64,
    sampling_ns: u64,
    sync_calls: u64,
    host_causality_edges: u64,
    hot_path_allocations: u64,
    deepseek_compressor_state_writes: u64,
    deepseek_compressed_kv_writes: u64,
    deepseek_indexer_state_writes: u64,
    deepseek_indexer_kv_writes: u64,
    deepseek_compressed_kv_attention_reads: u64,
    deepseek_compressed_kv_attention_slots_scanned: u64,
    deepseek_sparse_topk_selections: u64,
    deepseek_sparse_topk_slots_selected: u64,
    deepseek_sparse_topk_candidates_scored: u64,
    deepseek_sparse_topk_selection_hash: u64,
    deepseek_v3_grouped_router_selections: u64,
    deepseek_v4_bias_router_selections: u64,
    deepseek_v4_hash_router_selections: u64,
    deepseek_raw_attention_tokens_scanned: u64,
    deepseek_sparse_attention_output_hash: u64,
    cuda_footprint: CudaHfDecodeSequenceFootprint,
    cuda_device_total_memory_bytes: Option<usize>,
    cuda_device_free_memory_bytes: Option<usize>,
    cuda_fits_device_free_memory: Option<bool>,
}

impl CudaDecodeCounters {
    pub(super) fn record_sequence(&mut self, cuda: &CudaHfDecodeSequenceSummary) {
        self.resident_weight_bytes += cuda.resident_weight_bytes;
        self.resident_kv_bytes += cuda.resident_kv_bytes;
        self.kv_tokens = self.kv_tokens.max(cuda.kv_tokens);
        self.h2d_bytes += cuda.h2d_bytes;
        self.d2h_bytes += cuda.d2h_bytes;
        self.graph_replays += cuda.graph_replays;
        self.graph_nodes = self.graph_nodes.max(cuda.graph_nodes);
        self.graph_launches += cuda.graph_launches;
        self.graph_captures += cuda.graph_captures;
        self.graph_cache_hits += cuda.graph_cache_hits;
        self.kernel_launches += cuda.kernel_launches;
        self.experimental_rt_selector_launches += cuda.experimental_rt_selector_launches;
        if cuda.experimental_rt_sparse_attention_active {
            self.experimental_rt_sparse_attention_chunks = self
                .experimental_rt_sparse_attention_chunks
                .saturating_add(1);
        }
        self.experimental_rt_dense_attention_chunks = self
            .experimental_rt_dense_attention_chunks
            .max(cuda.experimental_rt_dense_attention_chunks);
        self.experimental_rt_attention_chunks = self
            .experimental_rt_attention_chunks
            .max(cuda.experimental_rt_attention_chunks);
        self.projection_ns += cuda.projection_ns;
        self.qkv_projection_ns += cuda.qkv_projection_ns;
        self.attention_output_projection_ns += cuda.attention_output_projection_ns;
        self.gate_up_projection_ns += cuda.gate_up_projection_ns;
        self.down_projection_ns += cuda.down_projection_ns;
        self.lm_head_projection_ns += cuda.lm_head_projection_ns;
        self.attention_ns += cuda.attention_ns;
        self.mlp_ns += cuda.mlp_ns;
        self.norm_ns += cuda.norm_ns;
        self.sampling_ns += cuda.sampling_ns;
        self.sync_calls += cuda.sync_calls;
        self.host_causality_edges += cuda.host_causality_edges;
        self.hot_path_allocations += cuda.hot_path_allocations;
        self.deepseek_compressor_state_writes += cuda.deepseek_compressor_state_writes;
        self.deepseek_compressed_kv_writes += cuda.deepseek_compressed_kv_writes;
        self.deepseek_indexer_state_writes += cuda.deepseek_indexer_state_writes;
        self.deepseek_indexer_kv_writes += cuda.deepseek_indexer_kv_writes;
        self.deepseek_compressed_kv_attention_reads += cuda.deepseek_compressed_kv_attention_reads;
        self.deepseek_compressed_kv_attention_slots_scanned +=
            cuda.deepseek_compressed_kv_attention_slots_scanned;
        self.deepseek_sparse_topk_selections += cuda.deepseek_sparse_topk_selections;
        self.deepseek_sparse_topk_slots_selected += cuda.deepseek_sparse_topk_slots_selected;
        self.deepseek_sparse_topk_candidates_scored += cuda.deepseek_sparse_topk_candidates_scored;
        self.deepseek_sparse_topk_selection_hash ^= cuda.deepseek_sparse_topk_selection_hash;
        self.deepseek_v3_grouped_router_selections += cuda.deepseek_v3_grouped_router_selections;
        self.deepseek_v4_bias_router_selections += cuda.deepseek_v4_bias_router_selections;
        self.deepseek_v4_hash_router_selections += cuda.deepseek_v4_hash_router_selections;
        self.deepseek_raw_attention_tokens_scanned += cuda.deepseek_raw_attention_tokens_scanned;
        self.deepseek_sparse_attention_output_hash ^= cuda.deepseek_sparse_attention_output_hash;
        self.cuda_footprint = cuda.planned_footprint;
        self.cuda_device_total_memory_bytes = cuda.device_total_memory_bytes;
        self.cuda_device_free_memory_bytes = cuda.device_free_memory_bytes;
        self.cuda_fits_device_free_memory = cuda.fits_device_free_memory;
    }
}

pub(super) struct DecodeParts {
    steps_requested: usize,
    tokens: Vec<TokenId>,
    expected_tokens: Vec<TokenId>,
    reference_mode: &'static str,
    reference_verified: bool,
    ledgers: Vec<TokenLedger>,
    resident_weights: HfCudaResidentWeightSummary,
}

impl DecodeParts {
    pub(super) fn new(
        steps_requested: usize,
        tokens: Vec<TokenId>,
        expected_tokens: Vec<TokenId>,
        reference_mode: &'static str,
        reference_verified: bool,
        ledgers: Vec<TokenLedger>,
    ) -> Self {
        Self {
            steps_requested,
            tokens,
            expected_tokens,
            reference_mode,
            reference_verified,
            ledgers,
            resident_weights: HfCudaResidentWeightSummary::default(),
        }
    }
}

pub(super) fn build_summary(
    status: SmokeStatus,
    parts: DecodeParts,
    cpu_ledgers: &[TokenLedger],
    counters: CudaDecodeCounters,
    error: Option<String>,
) -> HfCudaSeedDecodeSummary {
    let output_hash = hash_tokens(&parts.tokens);
    let expected_hash = hash_tokens(&parts.expected_tokens);
    let critical_paths = critical_paths(&parts.ledgers);
    HfCudaSeedDecodeSummary {
        status,
        steps_requested: parts.steps_requested,
        parity: parts.reference_verified && parts.tokens == parts.expected_tokens,
        reference_mode: parts.reference_mode,
        reference_verified: parts.reference_verified,
        ledger_count: parts.ledgers.len() as u64,
        device_events: event_count(&parts.ledgers, LedgerEventKind::DeviceActivity),
        copy_events: event_count(&parts.ledgers, LedgerEventKind::Copy),
        hard_syncs: sync_count(&parts.ledgers, SyncClass::HardSync),
        soft_visibility_syncs: sync_count(&parts.ledgers, SyncClass::SoftVisibilitySync),
        execution_decisions: execution_decisions(&parts.ledgers),
        resident_weight_bytes: counters.resident_weight_bytes,
        cuda_footprint: counters.cuda_footprint,
        cuda_device_total_memory_bytes: counters.cuda_device_total_memory_bytes,
        cuda_device_free_memory_bytes: counters.cuda_device_free_memory_bytes,
        cuda_fits_device_free_memory: counters.cuda_fits_device_free_memory,
        resident_kv_bytes: counters.resident_kv_bytes,
        kv_tokens: counters.kv_tokens,
        h2d_bytes: counters.h2d_bytes,
        d2h_bytes: counters.d2h_bytes,
        graph_replays: counters.graph_replays,
        graph_nodes: counters.graph_nodes,
        graph_launches: counters.graph_launches,
        graph_captures: counters.graph_captures,
        graph_cache_hits: counters.graph_cache_hits,
        graph_replay_events: event_count(&parts.ledgers, LedgerEventKind::GraphReplay),
        kernel_launches: counters.kernel_launches,
        experimental_rt_selector_launches: counters.experimental_rt_selector_launches,
        experimental_rt_sparse_attention_chunks: counters.experimental_rt_sparse_attention_chunks,
        experimental_rt_dense_attention_chunks: counters.experimental_rt_dense_attention_chunks,
        experimental_rt_attention_chunks: counters.experimental_rt_attention_chunks,
        projection_ns: counters.projection_ns,
        qkv_projection_ns: counters.qkv_projection_ns,
        attention_output_projection_ns: counters.attention_output_projection_ns,
        gate_up_projection_ns: counters.gate_up_projection_ns,
        down_projection_ns: counters.down_projection_ns,
        lm_head_projection_ns: counters.lm_head_projection_ns,
        attention_ns: counters.attention_ns,
        mlp_ns: counters.mlp_ns,
        norm_ns: counters.norm_ns,
        sampling_ns: counters.sampling_ns,
        sync_calls: counters.sync_calls,
        host_causality_edges: counters.host_causality_edges,
        hot_path_allocations: counters.hot_path_allocations
            + hot_path_allocations(&parts.ledgers)
            + hot_path_allocations(cpu_ledgers),
        deepseek_compressor_state_writes: counters.deepseek_compressor_state_writes,
        deepseek_compressed_kv_writes: counters.deepseek_compressed_kv_writes,
        deepseek_indexer_state_writes: counters.deepseek_indexer_state_writes,
        deepseek_indexer_kv_writes: counters.deepseek_indexer_kv_writes,
        deepseek_compressed_kv_attention_reads: counters.deepseek_compressed_kv_attention_reads,
        deepseek_compressed_kv_attention_slots_scanned: counters
            .deepseek_compressed_kv_attention_slots_scanned,
        deepseek_sparse_topk_selections: counters.deepseek_sparse_topk_selections,
        deepseek_sparse_topk_slots_selected: counters.deepseek_sparse_topk_slots_selected,
        deepseek_sparse_topk_candidates_scored: counters.deepseek_sparse_topk_candidates_scored,
        deepseek_sparse_topk_selection_hash: counters.deepseek_sparse_topk_selection_hash,
        deepseek_v3_grouped_router_selections: counters.deepseek_v3_grouped_router_selections,
        deepseek_v4_bias_router_selections: counters.deepseek_v4_bias_router_selections,
        deepseek_v4_hash_router_selections: counters.deepseek_v4_hash_router_selections,
        deepseek_raw_attention_tokens_scanned: counters.deepseek_raw_attention_tokens_scanned,
        deepseek_sparse_attention_output_hash: counters.deepseek_sparse_attention_output_hash,
        output_hash,
        expected_hash,
        resident_weights: parts.resident_weights,
        critical_paths,
        token_ledgers: parts.ledgers,
        tokens: parts.tokens,
        expected_tokens: parts.expected_tokens,
        error,
    }
}
