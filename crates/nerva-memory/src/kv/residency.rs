use nerva_core::types::block::ResidencyState;
use nerva_core::types::error::{NervaError, Result};
use nerva_core::types::id::{AllocationId, DeviceOrdinal, ResidentBlockId};
use nerva_core::types::memory::MemoryTier;
use nerva_core::types::ownership::ExecutionOwner;
use nerva_ledger::types::decision::{CandidateCost, ResidencyDecision};
use nerva_ledger::types::event::{LedgerEvent, LedgerEventKind};
use nerva_ledger::types::metric::MetricSource;
use nerva_ledger::types::token::TokenLedger;

use crate::kv::page::KvPageDescriptor;
use crate::kv::pool::KvPagePool;
use crate::registry::BlockRegistry;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvResidencyPolicy {
    pub hot_page_limit: usize,
    pub prefetch_distance: u64,
    pub evict_after_idle: u64,
}

impl KvResidencyPolicy {
    pub const fn new(hot_page_limit: usize, prefetch_distance: u64, evict_after_idle: u64) -> Self {
        Self {
            hot_page_limit,
            prefetch_distance,
            evict_after_idle,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KvResidencyAction {
    KeepHot,
    PrefetchToHot,
    KeepWarm,
    DemoteToWarm,
    EvictCold,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct KvResidencyPlanEntry {
    pub page_index: u32,
    pub block_id: ResidentBlockId,
    pub bytes: usize,
    pub old_tier: MemoryTier,
    pub new_tier: MemoryTier,
    pub action: KvResidencyAction,
    pub reason: &'static str,
    pub predicted_visible_ns: u64,
}

impl KvResidencyPlanEntry {
    pub fn changes_tier(self) -> bool {
        self.old_tier != self.new_tier
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KvResidencyPlan {
    pub entries: Vec<KvResidencyPlanEntry>,
}

impl KvResidencyPlan {
    pub fn record_decisions_to_ledger(&self, ledger: &mut TokenLedger) {
        for entry in &self.entries {
            ledger.record_residency_decision(ResidencyDecision {
                block_id: entry.block_id,
                old_tier: entry.old_tier,
                new_tier: entry.new_tier,
                executor_selected: ExecutionOwner::Gpu(DeviceOrdinal(0)),
                candidate_costs: vec![
                    CandidateCost::estimated("keep-current-tier", 0),
                    CandidateCost::estimated("planned-tier", entry.predicted_visible_ns),
                ],
                reason: entry.reason,
                predicted_overlap_ns: 0,
                actual_visible_ns: None,
                metric_source: MetricSource::EstimatedModel,
            });
        }
    }

    pub fn record_events_to_ledger(&self, ledger: &mut TokenLedger) {
        for entry in &self.entries {
            match entry.action {
                KvResidencyAction::PrefetchToHot => {
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Prefetch,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_prefetch_scheduled",
                    });
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Copy,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_prefetch_copy",
                    });
                    record_visible_transfer_stall(ledger, entry);
                }
                KvResidencyAction::DemoteToWarm => {
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Eviction,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_demote_scheduled",
                    });
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Copy,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_demote_copy",
                    });
                    record_visible_transfer_stall(ledger, entry);
                }
                KvResidencyAction::EvictCold => {
                    ledger.record(LedgerEvent {
                        kind: LedgerEventKind::Eviction,
                        sync_class: None,
                        metric_source: MetricSource::EstimatedModel,
                        block_id: Some(entry.block_id),
                        from_tier: Some(entry.old_tier),
                        to_tier: Some(entry.new_tier),
                        bytes: entry.bytes,
                        latency_ns: 0,
                        label: "kv_cold_eviction",
                    });
                    if entry.changes_tier() {
                        ledger.record(LedgerEvent {
                            kind: LedgerEventKind::Copy,
                            sync_class: None,
                            metric_source: MetricSource::EstimatedModel,
                            block_id: Some(entry.block_id),
                            from_tier: Some(entry.old_tier),
                            to_tier: Some(entry.new_tier),
                            bytes: entry.bytes,
                            latency_ns: 0,
                            label: "kv_eviction_copy",
                        });
                    }
                    record_visible_transfer_stall(ledger, entry);
                }
                KvResidencyAction::KeepHot | KvResidencyAction::KeepWarm => {}
            }
        }
    }

    pub fn record_to_ledger(&self, ledger: &mut TokenLedger) {
        self.record_decisions_to_ledger(ledger);
        self.record_events_to_ledger(ledger);
    }

    pub fn apply(&self, registry: &mut BlockRegistry) -> Result<()> {
        for entry in &self.entries {
            if entry.changes_tier() {
                registry.move_block(
                    entry.block_id,
                    entry.new_tier,
                    AllocationId(entry.block_id.0),
                    0,
                )?;
            }
            match entry.action {
                KvResidencyAction::KeepHot
                | KvResidencyAction::PrefetchToHot
                | KvResidencyAction::KeepWarm => registry.mark_ready(entry.block_id)?,
                KvResidencyAction::DemoteToWarm => {
                    registry.transition(entry.block_id, ResidencyState::Draining)?
                }
                KvResidencyAction::EvictCold => {
                    registry.transition(entry.block_id, ResidencyState::Evicting)?
                }
            }
        }
        Ok(())
    }

    pub fn action_count(&self, action: KvResidencyAction) -> u64 {
        self.entries
            .iter()
            .filter(|entry| entry.action == action)
            .count() as u64
    }

    pub fn changed_bytes(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.changes_tier())
            .map(|entry| entry.bytes)
            .sum()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct KvPagePriority {
    page_index: u32,
    pinned: bool,
    distance: u64,
    last_use: u64,
}

pub struct KvResidencyPlanner;

impl KvResidencyPlanner {
    pub fn plan(
        pool: &KvPagePool,
        registry: &BlockRegistry,
        current_step: u64,
        policy: KvResidencyPolicy,
    ) -> Result<KvResidencyPlan> {
        let mut hot_candidates = Vec::new();
        for page in pool.pages() {
            if page.is_free && page.prefix_key.is_none() {
                continue;
            }
            let pinned = page.ref_count > 0;
            let distance = page
                .next_use
                .map(|next_use| next_use.saturating_sub(current_step))
                .unwrap_or(u64::MAX);
            if pinned || distance <= policy.prefetch_distance {
                hot_candidates.push(KvPagePriority {
                    page_index: page.page_index,
                    pinned,
                    distance,
                    last_use: page.last_use,
                });
            }
        }
        hot_candidates.sort_by_key(|candidate| {
            (
                !candidate.pinned,
                candidate.distance,
                core::cmp::Reverse(candidate.last_use),
                candidate.page_index,
            )
        });
        let pinned_count = hot_candidates
            .iter()
            .filter(|candidate| candidate.pinned)
            .count();
        let speculative_budget = policy.hot_page_limit.saturating_sub(pinned_count);
        let mut speculative_taken = 0usize;
        let hot_pages = hot_candidates
            .into_iter()
            .filter_map(|candidate| {
                if candidate.pinned {
                    Some(candidate.page_index)
                } else if speculative_taken < speculative_budget {
                    speculative_taken += 1;
                    Some(candidate.page_index)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut entries = Vec::new();
        for page in pool.pages() {
            if page.is_free && page.prefix_key.is_none() {
                continue;
            }
            let old_tier = registry
                .block(page.block_id)
                .ok_or_else(|| NervaError::InvalidArgument {
                    reason: format!("KV page {} references missing block", page.page_index),
                })?
                .tier;
            let wants_hot = hot_pages.contains(&page.page_index);
            let idle = current_step.saturating_sub(page.last_use);
            let entry = if wants_hot && old_tier == MemoryTier::Vram {
                KvResidencyPlanEntry {
                    page_index: page.page_index,
                    block_id: page.block_id,
                    bytes: page_token_bytes(page),
                    old_tier,
                    new_tier: MemoryTier::Vram,
                    action: KvResidencyAction::KeepHot,
                    reason: "KV page is already hot and needed soon",
                    predicted_visible_ns: 0,
                }
            } else if wants_hot {
                KvResidencyPlanEntry {
                    page_index: page.page_index,
                    block_id: page.block_id,
                    bytes: page_token_bytes(page),
                    old_tier,
                    new_tier: MemoryTier::Vram,
                    action: KvResidencyAction::PrefetchToHot,
                    reason: "KV page next use is within prefetch window",
                    predicted_visible_ns: transfer_cost_ns(
                        page_token_bytes(page),
                        old_tier,
                        MemoryTier::Vram,
                    ),
                }
            } else if page.ref_count == 0 && idle >= policy.evict_after_idle {
                KvResidencyPlanEntry {
                    page_index: page.page_index,
                    block_id: page.block_id,
                    bytes: page_token_bytes(page),
                    old_tier,
                    new_tier: MemoryTier::Dram,
                    action: KvResidencyAction::EvictCold,
                    reason: "KV page is idle beyond eviction threshold",
                    predicted_visible_ns: transfer_cost_ns(
                        page_token_bytes(page),
                        old_tier,
                        MemoryTier::Dram,
                    ),
                }
            } else if old_tier == MemoryTier::Vram {
                KvResidencyPlanEntry {
                    page_index: page.page_index,
                    block_id: page.block_id,
                    bytes: page_token_bytes(page),
                    old_tier,
                    new_tier: MemoryTier::Dram,
                    action: KvResidencyAction::DemoteToWarm,
                    reason: "KV page is outside hot budget",
                    predicted_visible_ns: transfer_cost_ns(
                        page_token_bytes(page),
                        old_tier,
                        MemoryTier::Dram,
                    ),
                }
            } else {
                KvResidencyPlanEntry {
                    page_index: page.page_index,
                    block_id: page.block_id,
                    bytes: page_token_bytes(page),
                    old_tier,
                    new_tier: old_tier,
                    action: KvResidencyAction::KeepWarm,
                    reason: "KV page remains warm",
                    predicted_visible_ns: 0,
                }
            };
            entries.push(entry);
        }
        Ok(KvResidencyPlan { entries })
    }
}

fn page_token_bytes(page: &KvPageDescriptor) -> usize {
    page.page_bytes
}

fn record_visible_transfer_stall(ledger: &mut TokenLedger, entry: &KvResidencyPlanEntry) {
    if entry.predicted_visible_ns == 0 {
        return;
    }
    ledger.record(LedgerEvent {
        kind: LedgerEventKind::Stall,
        sync_class: None,
        metric_source: MetricSource::EstimatedModel,
        block_id: Some(entry.block_id),
        from_tier: Some(entry.old_tier),
        to_tier: Some(entry.new_tier),
        bytes: entry.bytes,
        latency_ns: entry.predicted_visible_ns,
        label: "kv_visible_transfer_stall",
    });
}

fn transfer_cost_ns(bytes: usize, old_tier: MemoryTier, new_tier: MemoryTier) -> u64 {
    if old_tier == new_tier {
        0
    } else {
        100 + bytes as u64
    }
}
