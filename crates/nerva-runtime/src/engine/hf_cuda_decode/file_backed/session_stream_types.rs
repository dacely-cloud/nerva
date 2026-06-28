use nerva_core::types::dtype::DType;
use nerva_core::types::id::token::TokenId;
use nerva_cuda::decode::hf_sequence::session::summary::CudaHfDecodeSequenceSessionCreateSummary;
use nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use nerva_model::causal_lm::types::HfCausalLmStopReason;
use nerva_model::hf::metadata::HfModelMetadata;

use crate::engine::hf_cuda_decode::file_backed::projection_mode::HfCudaProjectionMode;
use crate::engine::hf_cuda_decode::summary::HfCudaSeedDecodeSummary;

pub struct HfCudaDeviceSessionStreamOutput {
    pub metadata: HfModelMetadata,
    pub dtype: DType,
    pub manifest_entries: usize,
    pub shard_plan_entries: usize,
    pub tensors_loaded: usize,
    pub bytes_loaded: usize,
    pub data_hash: u64,
    pub data_hash_available: bool,
    pub projection_mode: HfCudaProjectionMode,
    pub load_wall_ns: u64,
    pub prefill_wall_ns: u64,
    pub decode_wall_ns: u64,
    pub create: CudaHfDecodeSequenceSessionCreateSummary,
    pub start: CudaHfDecodeSequenceSummary,
    pub records: Vec<HfCudaDeviceSessionStreamRecord>,
    pub chunks: Vec<HfCudaSeedDecodeSummary>,
    pub tokens: Vec<TokenId>,
    pub queue: HfCudaHostOutputQueueSummary,
    pub stop_reason: HfCausalLmStopReason,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HfCudaDeviceSessionStreamRecord {
    pub token_index: u64,
    pub token: TokenId,
    pub chunk_index: usize,
    pub chunk_offset: usize,
    pub queue_slot: usize,
    pub host_visible_order: u64,
    pub device_authoritative: bool,
    pub host_causality_edge: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HfCudaHostOutputQueueSummary {
    pub capacity: usize,
    pub pushes: u64,
    pub drains: u64,
    pub high_watermark: usize,
    pub overflow_rejections: u64,
    pub host_causality_edges: u64,
}
