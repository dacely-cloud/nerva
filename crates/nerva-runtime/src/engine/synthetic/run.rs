use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::TokenId;
use nerva_core::types::id::{RequestId, ResidentBlockId, SequenceId};
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;

use crate::engine::runtime::Runtime;
use crate::engine::synthetic::config::SyntheticDecodeConfig;
use crate::engine::synthetic::summary::{SyntheticDecodeStatus, SyntheticDecodeSummary};
use crate::graph::layout::GraphKey;
use crate::token::ring::TokenInputSource;

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

struct SyntheticDecodeTotals {
    graph_replay_events: u64,
    kernel_events: u64,
    device_events: u64,
    copy_events: u64,
    host_wait_events: u64,
    soft_visibility_syncs: u64,
    device_timeline_active_ns: u64,
    device_timeline_idle_ns: u64,
    graph_replay_latency_ns: u64,
    device_latency_ns: u64,
    copy_latency_ns: u64,
    host_wait_latency_ns: u64,
    soft_visibility_sync_latency_ns: u64,
    estimated_events: u64,
    estimated_latency_ns: u64,
    total_latency_ns: u64,
    hot_path_allocations: u64,
    observed_tokens: u64,
    observed_token_hash: u64,
    stale_tokens: u64,
    extra_tokens: u64,
    mismatched_tokens: u64,
    host_causality_edges: u64,
    token_ring_slots_seen: Vec<bool>,
    token_ring_slots_touched: u64,
    token_ring_reuses: u64,
    token_ring_max_slot_version: u64,
}

impl SyntheticDecodeTotals {
    fn new(token_ring_capacity: usize) -> Self {
        Self {
            graph_replay_events: 0,
            kernel_events: 0,
            device_events: 0,
            copy_events: 0,
            host_wait_events: 0,
            soft_visibility_syncs: 0,
            device_timeline_active_ns: 0,
            device_timeline_idle_ns: 0,
            graph_replay_latency_ns: 0,
            device_latency_ns: 0,
            copy_latency_ns: 0,
            host_wait_latency_ns: 0,
            soft_visibility_sync_latency_ns: 0,
            estimated_events: 0,
            estimated_latency_ns: 0,
            total_latency_ns: 0,
            hot_path_allocations: 0,
            observed_tokens: 0,
            observed_token_hash: TOKEN_STREAM_HASH_SEED,
            stale_tokens: 0,
            extra_tokens: 0,
            mismatched_tokens: 0,
            host_causality_edges: 0,
            token_ring_slots_seen: vec![false; token_ring_capacity],
            token_ring_slots_touched: 0,
            token_ring_reuses: 0,
            token_ring_max_slot_version: 0,
        }
    }

    fn record_token(
        &mut self,
        runtime: &Runtime,
        output: &crate::token::step::StepOutput,
        token_index: u64,
        seed_token: TokenId,
    ) -> Result<()> {
        let token_graph_events = output.ledger.event_count(LedgerEventKind::GraphReplay);
        let token_device_events = output.ledger.event_count(LedgerEventKind::DeviceActivity);
        let token_kernel_events = output.ledger.event_count(LedgerEventKind::KernelLaunch)
            + token_graph_events
            + token_device_events;

        self.graph_replay_events += token_graph_events;
        self.kernel_events += token_kernel_events;
        self.device_events += token_device_events;
        self.copy_events += output.ledger.event_count(LedgerEventKind::Copy);
        self.host_wait_events += output.ledger.event_count(LedgerEventKind::Sync);
        self.soft_visibility_syncs += output.ledger.sync_count_for(SyncClass::SoftVisibilitySync);
        self.device_timeline_active_ns = self
            .device_timeline_active_ns
            .saturating_add(output.ledger.device_active_ns(runtime.config.device)?);
        self.device_timeline_idle_ns = self
            .device_timeline_idle_ns
            .saturating_add(output.ledger.device_idle_ns(runtime.config.device)?);
        self.graph_replay_latency_ns = self
            .graph_replay_latency_ns
            .saturating_add(output.ledger.latency_ns_for(LedgerEventKind::GraphReplay));
        self.device_latency_ns = self.device_latency_ns.saturating_add(
            output
                .ledger
                .latency_ns_for(LedgerEventKind::DeviceActivity),
        );
        self.copy_latency_ns = self
            .copy_latency_ns
            .saturating_add(output.ledger.latency_ns_for(LedgerEventKind::Copy));
        self.host_wait_latency_ns = self
            .host_wait_latency_ns
            .saturating_add(output.ledger.latency_ns_for(LedgerEventKind::Sync));
        self.soft_visibility_sync_latency_ns = self.soft_visibility_sync_latency_ns.saturating_add(
            output
                .ledger
                .sync_latency_ns_for(SyncClass::SoftVisibilitySync),
        );
        self.estimated_events = self.estimated_events.saturating_add(
            output
                .ledger
                .event_count_for_source(MetricSource::EstimatedModel),
        );
        self.estimated_latency_ns = self.estimated_latency_ns.saturating_add(
            output
                .ledger
                .latency_ns_for_source(MetricSource::EstimatedModel),
        );
        self.total_latency_ns = self
            .total_latency_ns
            .saturating_add(output.ledger.total_latency_ns());
        self.hot_path_allocations = self
            .hot_path_allocations
            .saturating_add(output.ledger.hot_path_allocations);
        self.observed_tokens = self.observed_tokens.saturating_add(1);
        self.observed_token_hash =
            hash_observed_token(self.observed_token_hash, output.token_index, output.token);
        self.record_token_ring_slot(output);
        self.record_token_correctness(output, token_index, seed_token);
        Ok(())
    }

    fn record_token_ring_slot(&mut self, output: &crate::token::step::StepOutput) {
        if let Some(seen) = self
            .token_ring_slots_seen
            .get_mut(output.device_token_ref.slot_index)
        {
            if !*seen {
                *seen = true;
                self.token_ring_slots_touched = self.token_ring_slots_touched.saturating_add(1);
            }
        }
        if output.device_token_ref.version > 1 {
            self.token_ring_reuses = self.token_ring_reuses.saturating_add(1);
        }
        self.token_ring_max_slot_version = self
            .token_ring_max_slot_version
            .max(output.device_token_ref.version);
    }

    fn record_token_correctness(
        &mut self,
        output: &crate::token::step::StepOutput,
        token_index: u64,
        seed_token: TokenId,
    ) {
        if output.token_index < token_index {
            self.stale_tokens = self.stale_tokens.saturating_add(1);
        } else if output.token_index > token_index {
            self.extra_tokens = self.extra_tokens.saturating_add(1);
        }
        let expected_token = TokenId(
            seed_token
                .0
                .wrapping_add((token_index as u32).wrapping_add(1)),
        );
        if output.token != expected_token {
            self.mismatched_tokens = self.mismatched_tokens.saturating_add(1);
        }
        if output.input_source == TokenInputSource::HostObservation {
            self.host_causality_edges = self.host_causality_edges.saturating_add(1);
        }
    }
}

const TOKEN_STREAM_HASH_SEED: u64 = 0xcbf2_9ce4_8422_2325;

fn hash_observed_token(current: u64, token_index: u64, token: TokenId) -> u64 {
    let mut hash = current ^ token_index.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hash = hash.rotate_left(13) ^ u64::from(token.0);
    hash.wrapping_mul(0xff51_afd7_ed55_8ccd)
}
