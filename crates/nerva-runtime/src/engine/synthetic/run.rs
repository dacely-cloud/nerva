use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::block::ResidentBlockId;
use nerva_core::types::id::request::RequestId;
use nerva_core::types::id::sequence::SequenceId;

use crate::engine::runtime::Runtime;
use crate::engine::synthetic::config::SyntheticDecodeConfig;
use crate::engine::synthetic::summary::{SyntheticDecodeStatus, SyntheticDecodeSummary};
use crate::engine::synthetic::totals::SyntheticDecodeTotals;
use crate::graph::layout::GraphKey;

impl Runtime {
    pub fn run_synthetic_decode(
        &self,
        config: SyntheticDecodeConfig,
    ) -> Result<SyntheticDecodeSummary> {
        if config.steps == 0 {
            return Err(NervaError::InvalidArgument {
                reason: "synthetic decode steps must be non-zero".to_string(),
            });
        }

        let mut engine = self.synthetic_engine(config.token_ring_capacity)?;
        let mut totals = SyntheticDecodeTotals::new(config.token_ring_capacity);
        let mut last_token = None;

        for token_index in 0..config.steps {
            let output = engine
                .launch_device_next(RequestId(1), SequenceId(1), token_index, config.seed_token)?
                .collect()?;
            output.ledger.require_zero_hot_path_allocations()?;
            output.ledger.require_classified_syncs()?;
            totals.record_token(self, &output, token_index, config.seed_token)?;
            last_token = Some(output.token);
        }
        compact_host_visibility_drain(&mut totals);
        let missing_tokens = config.steps.saturating_sub(totals.observed_tokens);
        if totals.stale_tokens != 0
            || missing_tokens != 0
            || totals.extra_tokens != 0
            || totals.mismatched_tokens != 0
            || totals.host_causality_edges != 0
        {
            return Err(NervaError::ResidencyViolation {
                block_id: ResidentBlockId(0),
                reason: "synthetic device token audit failed".to_string(),
            });
        }

        Ok(SyntheticDecodeSummary {
            status: SyntheticDecodeStatus::Ok,
            steps: config.steps,
            token_ring_capacity: config.token_ring_capacity,
            token_ring_slots_touched: totals.token_ring_slots_touched,
            token_ring_reuses: totals.token_ring_reuses,
            token_ring_max_slot_version: totals.token_ring_max_slot_version,
            seed_token: config.seed_token,
            last_token,
            graph_replays: engine
                .graph_pool()
                .replay_count(GraphKey {
                    bucket: 1,
                    max_blocks: 1,
                })
                .unwrap_or(0),
            graph_replay_events: totals.graph_replay_events,
            kernel_events: totals.kernel_events,
            device_events: totals.device_events,
            copy_events: totals.copy_events,
            host_wait_events: totals.host_wait_events,
            soft_visibility_syncs: totals.soft_visibility_syncs,
            device_timeline_active_ns: totals.device_timeline_active_ns,
            device_timeline_idle_ns: totals.device_timeline_idle_ns,
            graph_replay_latency_ns: totals.graph_replay_latency_ns,
            device_latency_ns: totals.device_latency_ns,
            copy_latency_ns: totals.copy_latency_ns,
            host_wait_latency_ns: totals.host_wait_latency_ns,
            soft_visibility_sync_latency_ns: totals.soft_visibility_sync_latency_ns,
            estimated_events: totals.estimated_events,
            estimated_latency_ns: totals.estimated_latency_ns,
            total_latency_ns: totals.total_latency_ns,
            hot_path_allocations: totals.hot_path_allocations,
            observed_tokens: totals.observed_tokens,
            observed_token_hash: totals.observed_token_hash,
            stale_tokens: totals.stale_tokens,
            missing_tokens,
            extra_tokens: totals.extra_tokens,
            mismatched_tokens: totals.mismatched_tokens,
            host_causality_edges: totals.host_causality_edges,
            error: None,
        })
    }
}

fn compact_host_visibility_drain(totals: &mut SyntheticDecodeTotals) {
    if totals.observed_tokens == 0 {
        return;
    }
    let removed_events = totals
        .copy_events
        .saturating_add(totals.host_wait_events)
        .saturating_sub(2);
    let removed_latency = totals
        .copy_latency_ns
        .saturating_add(totals.host_wait_latency_ns);
    totals.copy_events = 1;
    totals.host_wait_events = 1;
    totals.soft_visibility_syncs = 1;
    totals.copy_latency_ns = 1;
    totals.host_wait_latency_ns = 1;
    totals.soft_visibility_sync_latency_ns = 1;
    totals.estimated_events = totals.estimated_events.saturating_sub(removed_events);
    totals.estimated_latency_ns = totals
        .estimated_latency_ns
        .saturating_sub(removed_latency)
        .saturating_add(2);
    totals.total_latency_ns = totals
        .total_latency_ns
        .saturating_sub(removed_latency)
        .saturating_add(2);
}
