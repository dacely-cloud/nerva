use nerva_runtime::engine::hf_cuda_decode::file_backed::generate::HfCudaDeviceGenerateOutput;

#[derive(Clone, Debug, Default)]
pub(crate) struct DecodeStats {
    pub tokens: usize,
    pub draft_tokens: usize,
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
}

impl DecodeStats {
    pub(crate) fn from_output(output: &HfCudaDeviceGenerateOutput) -> Self {
        let mut latencies = Vec::new();
        let mut stats = Self {
            tokens: output.tokens().len(),
            ..Self::default()
        };
        for chunk in &output.stream.chunks {
            stats.draft_tokens += chunk.steps_requested;
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

    pub(crate) fn acceptance(&self) -> f64 {
        if self.draft_tokens == 0 {
            0.0
        } else {
            self.tokens as f64 / self.draft_tokens as f64
        }
    }
}

fn percentile(sorted: &[u64], fraction: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}
