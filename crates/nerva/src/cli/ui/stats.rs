use nerva_runtime::engine::hf_cuda_decode::file_backed::generate::HfCudaDeviceGenerateOutput;

#[derive(Clone, Debug, Default)]
pub(crate) struct DecodeStats {
    pub tokens: usize,
    pub wall_ns: u64,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub projection_ns: u64,
    pub attention_ns: u64,
    pub mlp_ns: u64,
    pub norm_ns: u64,
    pub sampling_ns: u64,
    pub graph_nodes: u64,
    pub graph_replays: u64,
    pub graph_cache_hits: u64,
    pub kernel_launches: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub sync_calls: u64,
    pub hot_path_allocations: u64,
    pub host_causality_edges: u64,
    pub deepseek_compressor_state_writes: u64,
    pub deepseek_compressed_kv_writes: u64,
    pub deepseek_indexer_state_writes: u64,
    pub deepseek_indexer_kv_writes: u64,
    pub deepseek_compressed_kv_attention_reads: u64,
    pub deepseek_compressed_kv_attention_slots_scanned: u64,
    pub deepseek_sparse_topk_selections: u64,
    pub deepseek_sparse_topk_slots_selected: u64,
    pub deepseek_sparse_topk_candidates_scored: u64,
    pub deepseek_v3_grouped_router_selections: u64,
    pub deepseek_v4_bias_router_selections: u64,
    pub deepseek_v4_hash_router_selections: u64,
    pub deepseek_raw_attention_tokens_scanned: u64,
}

impl DecodeStats {
    pub(crate) fn from_output(output: &HfCudaDeviceGenerateOutput) -> Self {
        let mut latencies = Vec::new();
        let mut stats = Self {
            tokens: output.tokens().len(),
            ..Self::default()
        };
        for chunk in &output.stream.chunks {
            stats.projection_ns += chunk.projection_ns;
            stats.attention_ns += chunk.attention_ns;
            stats.mlp_ns += chunk.mlp_ns;
            stats.norm_ns += chunk.norm_ns;
            stats.sampling_ns += chunk.sampling_ns;
            stats.graph_nodes += chunk.graph_nodes;
            stats.graph_replays += chunk.graph_replays;
            stats.graph_cache_hits += chunk.graph_cache_hits;
            stats.kernel_launches += chunk.kernel_launches;
            stats.h2d_bytes += chunk.h2d_bytes;
            stats.d2h_bytes += chunk.d2h_bytes;
            stats.sync_calls += chunk.sync_calls;
            stats.hot_path_allocations += chunk.hot_path_allocations;
            stats.host_causality_edges += chunk.host_causality_edges;
            stats.deepseek_compressor_state_writes += chunk.deepseek_compressor_state_writes;
            stats.deepseek_compressed_kv_writes += chunk.deepseek_compressed_kv_writes;
            stats.deepseek_indexer_state_writes += chunk.deepseek_indexer_state_writes;
            stats.deepseek_indexer_kv_writes += chunk.deepseek_indexer_kv_writes;
            stats.deepseek_compressed_kv_attention_reads +=
                chunk.deepseek_compressed_kv_attention_reads;
            stats.deepseek_compressed_kv_attention_slots_scanned +=
                chunk.deepseek_compressed_kv_attention_slots_scanned;
            stats.deepseek_sparse_topk_selections += chunk.deepseek_sparse_topk_selections;
            stats.deepseek_sparse_topk_slots_selected += chunk.deepseek_sparse_topk_slots_selected;
            stats.deepseek_sparse_topk_candidates_scored +=
                chunk.deepseek_sparse_topk_candidates_scored;
            stats.deepseek_v3_grouped_router_selections +=
                chunk.deepseek_v3_grouped_router_selections;
            stats.deepseek_v4_bias_router_selections += chunk.deepseek_v4_bias_router_selections;
            stats.deepseek_v4_hash_router_selections += chunk.deepseek_v4_hash_router_selections;
            stats.deepseek_raw_attention_tokens_scanned +=
                chunk.deepseek_raw_attention_tokens_scanned;
            for path in &chunk.critical_paths {
                stats.wall_ns += path.wall_latency_ns;
                latencies.push(path.wall_latency_ns);
            }
        }
        latencies.sort_unstable();
        stats.p50_ns = percentile(&latencies, 0.50);
        stats.p95_ns = percentile(&latencies, 0.95);
        stats.p99_ns = percentile(&latencies, 0.99);
        stats
    }

    pub(crate) fn mean_ns(&self) -> u64 {
        if self.tokens == 0 {
            0
        } else {
            self.wall_ns / self.tokens as u64
        }
    }

    pub(crate) fn has_deepseek_activity(&self) -> bool {
        self.deepseek_compressor_state_writes != 0
            || self.deepseek_compressed_kv_writes != 0
            || self.deepseek_indexer_state_writes != 0
            || self.deepseek_indexer_kv_writes != 0
            || self.deepseek_compressed_kv_attention_reads != 0
            || self.deepseek_compressed_kv_attention_slots_scanned != 0
            || self.deepseek_sparse_topk_selections != 0
            || self.deepseek_sparse_topk_slots_selected != 0
            || self.deepseek_sparse_topk_candidates_scored != 0
            || self.deepseek_v3_grouped_router_selections != 0
            || self.deepseek_v4_bias_router_selections != 0
            || self.deepseek_v4_hash_router_selections != 0
            || self.deepseek_raw_attention_tokens_scanned != 0
    }
}

fn percentile(sorted: &[u64], fraction: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}
