#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct KernelUdpLatencyStats {
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub total_ns: u64,
}

pub(crate) fn latency_stats(latencies: &[u64]) -> KernelUdpLatencyStats {
    let mut sorted = latencies.to_vec();
    sorted.sort_unstable();
    KernelUdpLatencyStats {
        p50_ns: percentile(&sorted, 50),
        p95_ns: percentile(&sorted, 95),
        p99_ns: percentile(&sorted, 99),
        total_ns: sorted.iter().sum(),
    }
}

pub(crate) fn bandwidth_bps(bytes: usize, elapsed_ns: u64) -> u64 {
    if elapsed_ns == 0 {
        0
    } else {
        ((bytes as u128)
            .saturating_mul(1_000_000_000)
            .saturating_div(elapsed_ns as u128))
        .min(u64::MAX as u128) as u64
    }
}

fn percentile(sorted: &[u64], percentile: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let max_index = sorted.len() - 1;
    let index = max_index.saturating_mul(percentile).div_ceil(100);
    sorted[index.min(max_index)]
}
