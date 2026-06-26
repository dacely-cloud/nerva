use nerva_core::types::error::Result;
use nerva_core::types::id::TokenId;
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;

use crate::engine::runtime::Runtime;
use crate::engine::synthetic::hash::{TOKEN_STREAM_HASH_SEED, hash_observed_token};
use crate::token::ring::TokenInputSource;
use crate::token::step::StepOutput;

pub(crate) struct SyntheticDecodeTotals {
    pub(crate) graph_replay_events: u64,
    pub(crate) kernel_events: u64,
    pub(crate) device_events: u64,
    pub(crate) copy_events: u64,
    pub(crate) host_wait_events: u64,
    pub(crate) soft_visibility_syncs: u64,
    pub(crate) device_timeline_active_ns: u64,
    pub(crate) device_timeline_idle_ns: u64,
    pub(crate) graph_replay_latency_ns: u64,
    pub(crate) device_latency_ns: u64,
    pub(crate) copy_latency_ns: u64,
    pub(crate) host_wait_latency_ns: u64,
    pub(crate) soft_visibility_sync_latency_ns: u64,
    pub(crate) estimated_events: u64,
    pub(crate) estimated_latency_ns: u64,
    pub(crate) total_latency_ns: u64,
    pub(crate) hot_path_allocations: u64,
    pub(crate) observed_tokens: u64,
    pub(crate) observed_token_hash: u64,
    pub(crate) stale_tokens: u64,
    pub(crate) extra_tokens: u64,
    pub(crate) mismatched_tokens: u64,
    pub(crate) host_causality_edges: u64,
    token_ring_slots_seen: Vec<bool>,
    pub(crate) token_ring_slots_touched: u64,
    pub(crate) token_ring_reuses: u64,
    pub(crate) token_ring_max_slot_version: u64,
}

impl SyntheticDecodeTotals {
    pub(crate) fn new(token_ring_capacity: usize) -> Self {
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

    pub(crate) fn record_token(
        &mut self,
        runtime: &Runtime,
        output: &StepOutput,
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

    fn record_token_ring_slot(&mut self, output: &StepOutput) {
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
        output: &StepOutput,
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
