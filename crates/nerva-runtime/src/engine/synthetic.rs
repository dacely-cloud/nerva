use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::TokenId;
use nerva_core::types::id::{RequestId, ResidentBlockId, SequenceId};
use nerva_ledger::types::event::LedgerEventKind;
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::sync::SyncClass;

use crate::engine::runtime::Runtime;
use crate::graph::layout::GraphKey;
use crate::token::ring::TokenInputSource;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SyntheticDecodeConfig {
    pub steps: u64,
    pub token_ring_capacity: usize,
    pub seed_token: TokenId,
}

impl SyntheticDecodeConfig {
    pub const fn new(steps: u64, token_ring_capacity: usize, seed_token: TokenId) -> Self {
        Self {
            steps,
            token_ring_capacity,
            seed_token,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SyntheticDecodeStatus {
    Ok,
    Failed,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SyntheticDecodeSummary {
    pub status: SyntheticDecodeStatus,
    pub steps: u64,
    pub token_ring_capacity: usize,
    pub token_ring_slots_touched: u64,
    pub token_ring_reuses: u64,
    pub token_ring_max_slot_version: u64,
    pub seed_token: TokenId,
    pub last_token: Option<TokenId>,
    pub graph_replays: u64,
    pub graph_replay_events: u64,
    pub kernel_events: u64,
    pub device_events: u64,
    pub copy_events: u64,
    pub host_wait_events: u64,
    pub soft_visibility_syncs: u64,
    pub device_timeline_active_ns: u64,
    pub device_timeline_idle_ns: u64,
    pub graph_replay_latency_ns: u64,
    pub device_latency_ns: u64,
    pub copy_latency_ns: u64,
    pub host_wait_latency_ns: u64,
    pub soft_visibility_sync_latency_ns: u64,
    pub estimated_events: u64,
    pub estimated_latency_ns: u64,
    pub total_latency_ns: u64,
    pub hot_path_allocations: u64,
    pub observed_tokens: u64,
    pub observed_token_hash: u64,
    pub stale_tokens: u64,
    pub missing_tokens: u64,
    pub extra_tokens: u64,
    pub mismatched_tokens: u64,
    pub host_causality_edges: u64,
    pub error: Option<&'static str>,
}

impl SyntheticDecodeSummary {
    pub fn to_json(self) -> String {
        let status = match self.status {
            SyntheticDecodeStatus::Ok => "ok",
            SyntheticDecodeStatus::Failed => "failed",
        };
        format!(
            "{{\"status\":\"{}\",\"steps\":{},\"token_ring_capacity\":{},\"token_ring_slots_touched\":{},\"token_ring_reuses\":{},\"token_ring_max_slot_version\":{},\"seed_token\":{},\"last_token\":{},\"graph_replays\":{},\"graph_replay_events\":{},\"kernel_events\":{},\"device_events\":{},\"copy_events\":{},\"host_wait_events\":{},\"soft_visibility_syncs\":{},\"device_timeline_active_ns\":{},\"device_timeline_idle_ns\":{},\"graph_replay_latency_ns\":{},\"device_latency_ns\":{},\"copy_latency_ns\":{},\"host_wait_latency_ns\":{},\"soft_visibility_sync_latency_ns\":{},\"estimated_events\":{},\"estimated_latency_ns\":{},\"total_latency_ns\":{},\"hot_path_allocations\":{},\"observed_tokens\":{},\"observed_token_hash\":{},\"stale_tokens\":{},\"missing_tokens\":{},\"extra_tokens\":{},\"mismatched_tokens\":{},\"host_causality_edges\":{},\"error\":{}}}",
            status,
            self.steps,
            self.token_ring_capacity,
            self.token_ring_slots_touched,
            self.token_ring_reuses,
            self.token_ring_max_slot_version,
            self.seed_token.0,
            json_opt_token(self.last_token),
            self.graph_replays,
            self.graph_replay_events,
            self.kernel_events,
            self.device_events,
            self.copy_events,
            self.host_wait_events,
            self.soft_visibility_syncs,
            self.device_timeline_active_ns,
            self.device_timeline_idle_ns,
            self.graph_replay_latency_ns,
            self.device_latency_ns,
            self.copy_latency_ns,
            self.host_wait_latency_ns,
            self.soft_visibility_sync_latency_ns,
            self.estimated_events,
            self.estimated_latency_ns,
            self.total_latency_ns,
            self.hot_path_allocations,
            self.observed_tokens,
            self.observed_token_hash,
            self.stale_tokens,
            self.missing_tokens,
            self.extra_tokens,
            self.mismatched_tokens,
            self.host_causality_edges,
            json_opt_static_str(self.error),
        )
    }
}

fn json_opt_token(value: Option<TokenId>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.0.to_string())
}

fn json_opt_static_str(value: Option<&'static str>) -> String {
    value.map_or_else(|| "null".to_string(), |value| format!("\"{value}\""))
}

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
        let mut last_token = None;
        let mut graph_replay_events: u64 = 0;
        let mut kernel_events: u64 = 0;
        let mut device_events: u64 = 0;
        let mut copy_events: u64 = 0;
        let mut host_wait_events: u64 = 0;
        let mut soft_visibility_syncs: u64 = 0;
        let mut device_timeline_active_ns: u64 = 0;
        let mut device_timeline_idle_ns: u64 = 0;
        let mut graph_replay_latency_ns: u64 = 0;
        let mut device_latency_ns: u64 = 0;
        let mut copy_latency_ns: u64 = 0;
        let mut host_wait_latency_ns: u64 = 0;
        let mut soft_visibility_sync_latency_ns: u64 = 0;
        let mut estimated_events: u64 = 0;
        let mut estimated_latency_ns: u64 = 0;
        let mut total_latency_ns: u64 = 0;
        let mut hot_path_allocations: u64 = 0;
        let mut observed_tokens: u64 = 0;
        let mut observed_token_hash: u64 = TOKEN_STREAM_HASH_SEED;
        let mut stale_tokens: u64 = 0;
        let mut extra_tokens: u64 = 0;
        let mut mismatched_tokens: u64 = 0;
        let mut host_causality_edges: u64 = 0;
        let mut token_ring_slots_seen = vec![false; config.token_ring_capacity];
        let mut token_ring_slots_touched: u64 = 0;
        let mut token_ring_reuses: u64 = 0;
        let mut token_ring_max_slot_version: u64 = 0;

        for token_index in 0..config.steps {
            let output = engine
                .launch_device_next(RequestId(1), SequenceId(1), token_index, config.seed_token)?
                .collect()?;
            output.ledger.require_zero_hot_path_allocations()?;
            output.ledger.require_classified_syncs()?;
            let token_graph_events = output.ledger.event_count(LedgerEventKind::GraphReplay);
            let token_device_events = output.ledger.event_count(LedgerEventKind::DeviceActivity);
            let token_kernel_events = output.ledger.event_count(LedgerEventKind::KernelLaunch)
                + token_graph_events
                + token_device_events;
            let token_copy_events = output.ledger.event_count(LedgerEventKind::Copy);
            let token_host_wait_events = output.ledger.event_count(LedgerEventKind::Sync);
            let token_soft_visibility_syncs =
                output.ledger.sync_count_for(SyncClass::SoftVisibilitySync);

            graph_replay_events += token_graph_events;
            kernel_events += token_kernel_events;
            device_events += token_device_events;
            copy_events += token_copy_events;
            host_wait_events += token_host_wait_events;
            soft_visibility_syncs += token_soft_visibility_syncs;
            device_timeline_active_ns = device_timeline_active_ns
                .saturating_add(output.ledger.device_active_ns(self.config.device)?);
            device_timeline_idle_ns = device_timeline_idle_ns
                .saturating_add(output.ledger.device_idle_ns(self.config.device)?);
            graph_replay_latency_ns = graph_replay_latency_ns
                .saturating_add(output.ledger.latency_ns_for(LedgerEventKind::GraphReplay));
            device_latency_ns = device_latency_ns.saturating_add(
                output
                    .ledger
                    .latency_ns_for(LedgerEventKind::DeviceActivity),
            );
            copy_latency_ns =
                copy_latency_ns.saturating_add(output.ledger.latency_ns_for(LedgerEventKind::Copy));
            host_wait_latency_ns = host_wait_latency_ns
                .saturating_add(output.ledger.latency_ns_for(LedgerEventKind::Sync));
            soft_visibility_sync_latency_ns = soft_visibility_sync_latency_ns.saturating_add(
                output
                    .ledger
                    .sync_latency_ns_for(SyncClass::SoftVisibilitySync),
            );
            estimated_events = estimated_events.saturating_add(
                output
                    .ledger
                    .event_count_for_source(MetricSource::EstimatedModel),
            );
            estimated_latency_ns = estimated_latency_ns.saturating_add(
                output
                    .ledger
                    .latency_ns_for_source(MetricSource::EstimatedModel),
            );
            total_latency_ns = total_latency_ns.saturating_add(output.ledger.total_latency_ns());
            hot_path_allocations =
                hot_path_allocations.saturating_add(output.ledger.hot_path_allocations);
            observed_tokens = observed_tokens.saturating_add(1);
            observed_token_hash =
                hash_observed_token(observed_token_hash, output.token_index, output.token);
            if let Some(seen) = token_ring_slots_seen.get_mut(output.device_token_ref.slot_index) {
                if !*seen {
                    *seen = true;
                    token_ring_slots_touched = token_ring_slots_touched.saturating_add(1);
                }
            }
            if output.device_token_ref.version > 1 {
                token_ring_reuses = token_ring_reuses.saturating_add(1);
            }
            token_ring_max_slot_version =
                token_ring_max_slot_version.max(output.device_token_ref.version);
            if output.token_index < token_index {
                stale_tokens = stale_tokens.saturating_add(1);
            } else if output.token_index > token_index {
                extra_tokens = extra_tokens.saturating_add(1);
            }
            let expected_token = TokenId(
                config
                    .seed_token
                    .0
                    .wrapping_add((token_index as u32).wrapping_add(1)),
            );
            if output.token != expected_token {
                mismatched_tokens = mismatched_tokens.saturating_add(1);
            }
            if output.input_source == TokenInputSource::HostObservation {
                host_causality_edges = host_causality_edges.saturating_add(1);
            }
            last_token = Some(output.token);
        }
        let missing_tokens = config.steps.saturating_sub(observed_tokens);
        if stale_tokens != 0
            || missing_tokens != 0
            || extra_tokens != 0
            || mismatched_tokens != 0
            || host_causality_edges != 0
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
            token_ring_slots_touched,
            token_ring_reuses,
            token_ring_max_slot_version,
            seed_token: config.seed_token,
            last_token,
            graph_replays: engine
                .graph_pool()
                .replay_count(GraphKey {
                    bucket: 1,
                    max_blocks: 1,
                })
                .unwrap_or(0),
            graph_replay_events,
            kernel_events,
            device_events,
            copy_events,
            host_wait_events,
            soft_visibility_syncs,
            device_timeline_active_ns,
            device_timeline_idle_ns,
            graph_replay_latency_ns,
            device_latency_ns,
            copy_latency_ns,
            host_wait_latency_ns,
            soft_visibility_sync_latency_ns,
            estimated_events,
            estimated_latency_ns,
            total_latency_ns,
            hot_path_allocations,
            observed_tokens,
            observed_token_hash,
            stale_tokens,
            missing_tokens,
            extra_tokens,
            mismatched_tokens,
            host_causality_edges,
            error: None,
        })
    }
}

const TOKEN_STREAM_HASH_SEED: u64 = 0xcbf2_9ce4_8422_2325;

fn hash_observed_token(current: u64, token_index: u64, token: TokenId) -> u64 {
    let mut hash = current ^ token_index.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hash = hash.rotate_left(13) ^ u64::from(token.0);
    hash.wrapping_mul(0xff51_afd7_ed55_8ccd)
}
