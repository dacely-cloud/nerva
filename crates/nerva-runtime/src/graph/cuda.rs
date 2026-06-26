use nerva_cuda::graph::CudaSyntheticGraphSummary;

pub fn cuda_synthetic_graph_smoke(
    steps: u32,
    ring_capacity: u32,
    seed_token: u32,
) -> CudaSyntheticGraphSummary {
    crate::engine::cuda::cuda_synthetic_graph_smoke(steps, ring_capacity, seed_token)
}
