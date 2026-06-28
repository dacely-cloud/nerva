use std::path::Path;

use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::token::TokenId;
use nerva_model::causal_lm::types::HfCausalLmStopReason;

use crate::engine::hf_cuda_decode::file_backed::progress::HfCudaDeviceSessionChunkProgress;
use crate::engine::hf_cuda_decode::file_backed::session_stream::run_hf_causal_lm_cuda_shard_backed_device_session_stream_with_progress;
use crate::engine::hf_cuda_decode::file_backed::session_stream_types::HfCudaDeviceSessionStreamOutput;
use crate::engine::runtime::Runtime;

pub struct HfCudaDeviceGenerateOutput {
    pub max_new_tokens: usize,
    pub stream: HfCudaDeviceSessionStreamOutput,
}

impl HfCudaDeviceGenerateOutput {
    pub fn tokens(&self) -> &[TokenId] {
        &self.stream.tokens
    }

    pub const fn stop_reason(&self) -> HfCausalLmStopReason {
        self.stream.stop_reason
    }
}

pub fn run_hf_causal_lm_cuda_shard_backed_device_generate(
    runtime: &Runtime,
    dir: impl AsRef<Path>,
    prompt_tokens: &[TokenId],
    max_context_tokens: usize,
    max_new_tokens: usize,
    queue_capacity: usize,
    compute_capability: Option<u32>,
) -> Result<HfCudaDeviceGenerateOutput> {
    run_hf_causal_lm_cuda_shard_backed_device_generate_with_progress(
        runtime,
        dir,
        prompt_tokens,
        max_context_tokens,
        max_new_tokens,
        queue_capacity,
        compute_capability,
        |_| {},
    )
}

pub fn run_hf_causal_lm_cuda_shard_backed_device_generate_with_progress<F>(
    runtime: &Runtime,
    dir: impl AsRef<Path>,
    prompt_tokens: &[TokenId],
    max_context_tokens: usize,
    max_new_tokens: usize,
    queue_capacity: usize,
    compute_capability: Option<u32>,
    progress: F,
) -> Result<HfCudaDeviceGenerateOutput>
where
    F: FnMut(HfCudaDeviceSessionChunkProgress),
{
    if max_new_tokens == 0 {
        return Err(NervaError::InvalidArgument {
            reason: "HF CUDA generate max_new_tokens must be non-zero".to_string(),
        });
    }
    let stream = run_hf_causal_lm_cuda_shard_backed_device_session_stream_with_progress(
        runtime,
        dir,
        prompt_tokens,
        max_context_tokens,
        1,
        max_new_tokens,
        queue_capacity,
        compute_capability,
        progress,
    )?;
    Ok(HfCudaDeviceGenerateOutput {
        max_new_tokens,
        stream,
    })
}
