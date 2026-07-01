use nerva_cuda::decode::hf_sequence::session::request::CudaHfDecodeSequenceBatchAdvanceSummary;
use nerva_cuda::decode::hf_sequence::session::stateful::CudaHfDecodeSequenceLoop;
use nerva_cuda::decode::hf_sequence::summary::CudaHfDecodeSequenceSummary;
use nerva_cuda::smoke::status::SmokeStatus;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaDecodeBatchAdvanceConfig {
    pub target_block_tokens: u32,
    pub min_block_tokens: u32,
}

impl CudaDecodeBatchAdvanceConfig {
    pub fn new(target_block_tokens: u32, min_block_tokens: u32) -> Self {
        Self {
            target_block_tokens: target_block_tokens.max(1),
            min_block_tokens: min_block_tokens.max(1),
        }
    }
}

impl Default for CudaDecodeBatchAdvanceConfig {
    fn default() -> Self {
        Self {
            target_block_tokens: 32,
            min_block_tokens: 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaDecodeBatchAdvanceMode {
    Batched,
    FallbackSequential { reason: &'static str },
    BatchFailed { reason: &'static str },
}

#[derive(Clone, Debug)]
pub struct CudaDecodeBatchAdvanceOutput {
    pub mode: CudaDecodeBatchAdvanceMode,
    pub tokens_by_loop: Vec<Vec<u32>>,
    pub batch: Option<CudaHfDecodeSequenceBatchAdvanceSummary>,
    pub sequential: Vec<CudaHfDecodeSequenceSummary>,
}

impl CudaDecodeBatchAdvanceOutput {
    pub fn observed_tokens(&self) -> usize {
        self.tokens_by_loop.iter().map(Vec::len).sum()
    }

    pub fn used_batched_projection(&self) -> bool {
        matches!(self.mode, CudaDecodeBatchAdvanceMode::Batched)
    }
}

pub fn advance_decode_loops_once(
    loops: &mut [&mut CudaHfDecodeSequenceLoop<'_>],
    config: CudaDecodeBatchAdvanceConfig,
) -> CudaDecodeBatchAdvanceOutput {
    if loops.is_empty() {
        return CudaDecodeBatchAdvanceOutput {
            mode: CudaDecodeBatchAdvanceMode::FallbackSequential { reason: "no_loops" },
            tokens_by_loop: Vec::new(),
            batch: None,
            sequential: Vec::new(),
        };
    }
    if loops.len() < config.min_block_tokens as usize {
        return advance_sequential_once(loops, "below_min_block_tokens", None);
    }

    let summary = CudaHfDecodeSequenceLoop::batch_advance_one(
        loops,
        config.target_block_tokens,
        config.min_block_tokens,
    );
    if successful_batch_advance(&summary) {
        return CudaDecodeBatchAdvanceOutput {
            mode: CudaDecodeBatchAdvanceMode::Batched,
            tokens_by_loop: batch_tokens_by_loop(loops.len(), &summary.tokens),
            batch: Some(summary),
            sequential: Vec::new(),
        };
    }
    if safe_to_fallback_after_batch_rejection(&summary) {
        let reason = summary.reason;
        return advance_sequential_once(loops, reason, Some(summary));
    }

    let reason = summary.reason;
    CudaDecodeBatchAdvanceOutput {
        mode: CudaDecodeBatchAdvanceMode::BatchFailed { reason },
        tokens_by_loop: vec![Vec::new(); loops.len()],
        batch: Some(summary),
        sequential: Vec::new(),
    }
}

pub fn advance_decode_loops_sequential_once(
    loops: &mut [&mut CudaHfDecodeSequenceLoop<'_>],
    reason: &'static str,
) -> CudaDecodeBatchAdvanceOutput {
    advance_sequential_once(loops, reason, None)
}

fn advance_sequential_once(
    loops: &mut [&mut CudaHfDecodeSequenceLoop<'_>],
    reason: &'static str,
    batch: Option<CudaHfDecodeSequenceBatchAdvanceSummary>,
) -> CudaDecodeBatchAdvanceOutput {
    let mut tokens_by_loop = Vec::with_capacity(loops.len());
    let mut sequential = Vec::with_capacity(loops.len());
    for loop_state in loops.iter_mut() {
        let summary = loop_state.advance(1);
        tokens_by_loop.push(summary.tokens.clone());
        sequential.push(summary);
    }
    CudaDecodeBatchAdvanceOutput {
        mode: CudaDecodeBatchAdvanceMode::FallbackSequential { reason },
        tokens_by_loop,
        batch,
        sequential,
    }
}

fn successful_batch_advance(summary: &CudaHfDecodeSequenceBatchAdvanceSummary) -> bool {
    summary.status == SmokeStatus::Ok
        && summary.exact
        && summary.observed_tokens > 0
        && summary.observed_tokens == summary.block_tokens
}

fn safe_to_fallback_after_batch_rejection(
    summary: &CudaHfDecodeSequenceBatchAdvanceSummary,
) -> bool {
    summary.status == SmokeStatus::Ok
        && !summary.exact
        && summary.observed_tokens == 0
        && matches!(
            summary.reason,
            "no_ready_sessions" | "shared_weights_unproven" | "insufficient_compatible_ready"
        )
}

fn batch_tokens_by_loop(loop_count: usize, tokens: &[u32]) -> Vec<Vec<u32>> {
    (0..loop_count)
        .map(|index| match tokens.get(index).copied() {
            Some(token) if token != u32::MAX => vec![token],
            _ => Vec::new(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use nerva_cuda::decode::hf_sequence::session::request::CudaHfDecodeSequenceBatchAdvanceSummary;
    use nerva_cuda::smoke::status::SmokeStatus;

    use super::{
        CudaDecodeBatchAdvanceConfig, CudaDecodeBatchAdvanceMode, advance_decode_loops_once,
        batch_tokens_by_loop, safe_to_fallback_after_batch_rejection, successful_batch_advance,
    };

    fn batch_summary(
        status: SmokeStatus,
        reason: &'static str,
        exact: bool,
        block_tokens: u32,
        observed_tokens: u32,
        tokens: Vec<u32>,
    ) -> CudaHfDecodeSequenceBatchAdvanceSummary {
        CudaHfDecodeSequenceBatchAdvanceSummary {
            status,
            reason,
            exact,
            requested_session_count: tokens.len() as u32,
            eligible_session_count: block_tokens,
            block_tokens,
            target_block_tokens: 8,
            min_block_tokens: 2,
            dtype: 1,
            layer_count: 1,
            observed_tokens,
            last_token: tokens
                .iter()
                .copied()
                .filter(|token| *token != u32::MAX)
                .last()
                .unwrap_or(0),
            observed_token_hash: 0,
            tokens,
            d2h_bytes: 0,
            projection_elapsed_ns: 0,
            qkv_elapsed_ns: 0,
            attention_output_elapsed_ns: 0,
            gate_up_elapsed_ns: 0,
            down_elapsed_ns: 0,
            lm_head_elapsed_ns: 0,
            pack_kernel_launches: 0,
            projection_kernel_launches: 0,
            scatter_kernel_launches: 0,
            dependency_kernel_launches: 0,
            experimental_rt_selector_launches: 0,
            sampling_kernel_launches: 0,
            sync_calls: 0,
            hot_path_allocations: 0,
            cuda_error: 0,
        }
    }

    #[test]
    fn batch_advance_config_clamps_zero() {
        let config = CudaDecodeBatchAdvanceConfig::new(0, 0);

        assert_eq!(config.target_block_tokens, 1);
        assert_eq!(config.min_block_tokens, 1);
    }

    #[test]
    fn empty_loop_batch_advance_is_sequential_fallback_without_cuda() {
        let mut loops = [];
        let output = advance_decode_loops_once(&mut loops, CudaDecodeBatchAdvanceConfig::default());

        assert_eq!(
            output.mode,
            CudaDecodeBatchAdvanceMode::FallbackSequential { reason: "no_loops" }
        );
        assert_eq!(output.observed_tokens(), 0);
        assert!(!output.used_batched_projection());
        assert!(output.batch.is_none());
        assert!(output.sequential.is_empty());
    }

    #[test]
    fn successful_batch_requires_exact_observed_block() {
        let ok = batch_summary(SmokeStatus::Ok, "ready", true, 2, 2, vec![7, 9]);
        let partial = batch_summary(SmokeStatus::Ok, "ready", true, 2, 1, vec![7, u32::MAX]);

        assert!(successful_batch_advance(&ok));
        assert!(!successful_batch_advance(&partial));
    }

    #[test]
    fn batch_tokens_preserve_original_loop_indexes() {
        let tokens = batch_tokens_by_loop(4, &[11, u32::MAX, 13]);

        assert_eq!(
            tokens,
            vec![vec![11], Vec::<u32>::new(), vec![13], Vec::new()]
        );
    }

    #[test]
    fn only_preflight_rejections_fall_back_after_batch_attempt() {
        let safe = batch_summary(
            SmokeStatus::Ok,
            "insufficient_compatible_ready",
            false,
            0,
            0,
            Vec::new(),
        );
        let unsafe_failure = batch_summary(
            SmokeStatus::Failed,
            "ready",
            false,
            2,
            0,
            vec![u32::MAX, u32::MAX],
        );

        assert!(safe_to_fallback_after_batch_rejection(&safe));
        assert!(!safe_to_fallback_after_batch_rejection(&unsafe_failure));
    }
}
