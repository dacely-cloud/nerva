use nerva_cuda::attention::probe::tiered_attention_smoke;
use nerva_cuda::attention::summary::CudaTieredAttentionSummary;
use nerva_cuda::block::probe::{loaded_tiny_block_smoke, tiny_block_smoke};
use nerva_cuda::block::summary::{CudaLoadedTinyBlockSummary, CudaTinyBlockSummary};
use nerva_cuda::decode::probe::tiny_decode_smoke;
use nerva_cuda::decode::summary::CudaTinyDecodeSummary;
use nerva_cuda::graph::probe::synthetic_graph_smoke;
use nerva_cuda::graph::summary::CudaSyntheticGraphSummary;
use nerva_cuda::sampler::probe::greedy_sampler_smoke;
use nerva_cuda::sampler::summary::CudaGreedySamplerSummary;

pub fn cuda_tiny_block_smoke() -> CudaTinyBlockSummary {
    tiny_block_smoke()
}

pub fn cuda_loaded_tiny_block_smoke() -> CudaLoadedTinyBlockSummary {
    loaded_tiny_block_smoke()
}

pub fn cuda_tiered_attention_smoke() -> CudaTieredAttentionSummary {
    tiered_attention_smoke()
}

pub fn cuda_greedy_sampler_smoke() -> CudaGreedySamplerSummary {
    greedy_sampler_smoke()
}

pub fn cuda_tiny_decode_smoke(
    steps: u32,
    ring_capacity: u32,
    seed_token: u32,
) -> CudaTinyDecodeSummary {
    tiny_decode_smoke(steps, ring_capacity, seed_token)
}

pub fn cuda_synthetic_graph_smoke(
    steps: u32,
    ring_capacity: u32,
    seed_token: u32,
) -> CudaSyntheticGraphSummary {
    synthetic_graph_smoke(steps, ring_capacity, seed_token)
}
