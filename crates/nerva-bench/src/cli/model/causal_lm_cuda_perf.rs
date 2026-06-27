use nerva_runtime::engine::hf_cuda_decode::file_backed::session_stream::HfCudaDeviceSessionStreamOutput;

pub(crate) fn stream_perf_json(output: &HfCudaDeviceSessionStreamOutput) -> String {
    let latencies = token_wall_latencies(output);
    let generated_tokens = output.tokens.len();
    let measured_wall_ns = latencies.iter().sum::<u64>();
    let tokens_per_second = throughput(generated_tokens, measured_wall_ns);
    let mean_ms = mean_ms(&latencies);
    let p50_ms = percentile_ms(latencies.clone(), 0.50);
    let p95_ms = percentile_ms(latencies.clone(), 0.95);
    let p99_ms = percentile_ms(latencies, 0.99);
    let profile_projection_ns = sum_projection_ns(output);
    let profile_attention_ns = sum_attention_ns(output);
    let profile_accounted_ns = sum_profile_accounted_ns(output);
    let replay_profile_ratio = ratio(measured_wall_ns, profile_accounted_ns);
    let graph_nodes = output
        .chunks
        .iter()
        .map(|chunk| chunk.graph_nodes)
        .sum::<u64>();
    let kernel_launches = output
        .chunks
        .iter()
        .map(|chunk| chunk.kernel_launches)
        .sum::<u64>();
    format!(
        "{{\"generated_tokens\":{},\"measured_wall_latency_ns\":{},\"timing_source\":\"replay_critical_path_gpu_events\",\"profile_bucket_source\":\"profile_pass_gpu_events\",\"tokens_per_second\":{},\"token_mean_ms\":{},\"token_p50_ms\":{},\"token_p95_ms\":{},\"token_p99_ms\":{},\"measured_replay_ns_per_token\":{},\"profile_accounted_ns_per_token\":{},\"profile_replay_ratio\":{},\"graph_nodes_per_token\":{},\"kernel_launches_per_token\":{},\"projection_ns_per_token\":{},\"profile_projection_ns_per_token\":{},\"attention_ns_per_token\":{},\"profile_attention_ns_per_token\":{},\"profile_mlp_ns_per_token\":{},\"profile_norm_ns_per_token\":{},\"profile_sampling_ns_per_token\":{},\"host_causality_edges\":{},\"hot_path_allocations\":{}}}",
        generated_tokens,
        measured_wall_ns,
        json_f64(tokens_per_second),
        json_f64(mean_ms),
        json_f64(p50_ms),
        json_f64(p95_ms),
        json_f64(p99_ms),
        json_f64(per_token(measured_wall_ns, generated_tokens)),
        json_f64(per_token(profile_accounted_ns, generated_tokens)),
        json_f64(replay_profile_ratio),
        json_f64(per_token(graph_nodes, generated_tokens)),
        json_f64(per_token(kernel_launches, generated_tokens)),
        json_f64(per_token(profile_projection_ns, generated_tokens)),
        json_f64(per_token(profile_projection_ns, generated_tokens)),
        json_f64(per_token(profile_attention_ns, generated_tokens)),
        json_f64(per_token(profile_attention_ns, generated_tokens)),
        json_f64(per_token(sum_mlp_ns(output), generated_tokens)),
        json_f64(per_token(sum_norm_ns(output), generated_tokens)),
        json_f64(per_token(sum_sampling_ns(output), generated_tokens)),
        output.queue.host_causality_edges,
        sum_hot_path_allocations(output),
    )
}

fn token_wall_latencies(output: &HfCudaDeviceSessionStreamOutput) -> Vec<u64> {
    output
        .chunks
        .iter()
        .flat_map(|chunk| chunk.critical_paths.iter())
        .map(|path| path.wall_latency_ns)
        .collect()
}

fn throughput(tokens: usize, wall_ns: u64) -> f64 {
    if tokens == 0 || wall_ns == 0 {
        0.0
    } else {
        tokens as f64 * 1_000_000_000.0 / wall_ns as f64
    }
}

fn mean_ms(latencies: &[u64]) -> f64 {
    if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<u64>() as f64 / latencies.len() as f64 / 1_000_000.0
    }
}

fn percentile_ms(mut latencies: Vec<u64>, percentile: f64) -> f64 {
    if latencies.is_empty() {
        return 0.0;
    }
    latencies.sort_unstable();
    let rank = (percentile * latencies.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(latencies.len() - 1);
    latencies[index] as f64 / 1_000_000.0
}

fn per_token(value: u64, tokens: usize) -> f64 {
    if tokens == 0 {
        0.0
    } else {
        value as f64 / tokens as f64
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn sum_profile_accounted_ns(output: &HfCudaDeviceSessionStreamOutput) -> u64 {
    output
        .chunks
        .iter()
        .map(|chunk| {
            chunk.projection_ns
                + chunk.attention_ns
                + chunk.mlp_ns
                + chunk.norm_ns
                + chunk.sampling_ns
        })
        .sum()
}

fn sum_projection_ns(output: &HfCudaDeviceSessionStreamOutput) -> u64 {
    output.chunks.iter().map(|chunk| chunk.projection_ns).sum()
}

fn sum_attention_ns(output: &HfCudaDeviceSessionStreamOutput) -> u64 {
    output.chunks.iter().map(|chunk| chunk.attention_ns).sum()
}

fn sum_mlp_ns(output: &HfCudaDeviceSessionStreamOutput) -> u64 {
    output.chunks.iter().map(|chunk| chunk.mlp_ns).sum()
}

fn sum_norm_ns(output: &HfCudaDeviceSessionStreamOutput) -> u64 {
    output.chunks.iter().map(|chunk| chunk.norm_ns).sum()
}

fn sum_sampling_ns(output: &HfCudaDeviceSessionStreamOutput) -> u64 {
    output.chunks.iter().map(|chunk| chunk.sampling_ns).sum()
}

fn sum_hot_path_allocations(output: &HfCudaDeviceSessionStreamOutput) -> u64 {
    output
        .chunks
        .iter()
        .map(|chunk| chunk.hot_path_allocations)
        .sum()
}

fn json_f64(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.6}")
    } else {
        "null".to_string()
    }
}
