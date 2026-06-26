use nerva_cuda::block::{
    CudaLoadedTinyBlockSummary, CudaTinyBlockSummary, loaded_tiny_block_smoke, tiny_block_smoke,
};
use nerva_cuda::graph::{CudaSyntheticGraphSummary, synthetic_graph_smoke};

pub fn cuda_tiny_block_smoke() -> CudaTinyBlockSummary {
    tiny_block_smoke()
}

pub fn cuda_loaded_tiny_block_smoke() -> CudaLoadedTinyBlockSummary {
    loaded_tiny_block_smoke()
}

pub fn cuda_synthetic_graph_smoke(
    steps: u32,
    ring_capacity: u32,
    seed_token: u32,
) -> CudaSyntheticGraphSummary {
    synthetic_graph_smoke(steps, ring_capacity, seed_token)
}
